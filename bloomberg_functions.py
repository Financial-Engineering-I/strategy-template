################################################################################
##### For Bloomberg ------------------------------------------------------------
##### Can't use this if you're on a Mac :(
################################################################################
from __future__ import print_function
from __future__ import absolute_import

from optparse import OptionParser

import os
import platform as plat
import sys
if sys.version_info >= (3, 8) and plat.system().lower() == "windows":
    # pylint: disable=no-member
    with os.add_dll_directory(os.getenv('BLPAPI_LIBDIR')):
        import blpapi
else:
    import blpapi
from utils import date_to_str
import pandas as pd

def parseCmdLine():
    parser = OptionParser(description="Retrieve reference data.")
    parser.add_option("-a",
                      "--ip",
                      dest="host",
                      help="server name or IP (default: %default)",
                      metavar="ipAddress",
                      default="localhost")
    parser.add_option("-p",
                      dest="port",
                      type="int",
                      help="server port (default: %default)",
                      metavar="tcpPort",
                      default=8194)

    (options, args) = parser.parse_args()

    return options

def req_historical_data(bbg_identifier, startDate, endDate):


    # Recast start & end dates in Bloomberg's format
    startDate = date_to_str(startDate, "%Y%m%d")
    endDate = date_to_str(endDate, "%Y%m%d")

    if(pd.to_datetime(startDate) >= pd.to_datetime(endDate)):
        sys.exit(
            "in req_historical_data in 'bloomberg_functions.py': " + \
            "specified startDate is later than endDate!"
        )

    # First, check to see if there is already a local .p data file with the
    # data you need for bbg_identifier. If it's not there, create it.
    if not os.path.isdir("bbg_data"):
        os.makedirs("bbg_data")
        print("created the 'bbg_data' folder.")

    if (bbg_identifier + ".csv") in os.listdir("bbg_data"):
        old_bbg_data = pd.read_csv("bbg_data/" + bbg_identifier + ".csv")

        first_old = pd.to_datetime(min(old_bbg_data['Date'])).date()
        last_old  = pd.to_datetime(max(old_bbg_data['Date'])).date()

        first_new = pd.to_datetime(startDate).date()
        last_new  = pd.to_datetime(endDate).date()

        if first_old <= first_new and last_old >= last_new:
            # Don't need to make a query; have all data we need.
            histdata = old_bbg_data[[
                (pd.to_datetime(x).date() <= last_new) & (
                        pd.to_datetime(x).date() >= first_new
                ) for x in old_bbg_data['Date']
            ]]
            histdata.reset_index(drop=True, inplace=True)
            return histdata

        if first_old > first_new and last_old < last_new:
            # do nothing for now, just requery the bigger dataset. Can refine
            # this case later.
            print(
                "overwriting old data with date range: " + startDate + \
                " to " + endDate
            )
        else:
            if first_new < first_old:
                endDate = date_to_str(first_old, "%Y%m%d")
            else:
                startDate = date_to_str(last_old, "%Y%m%d")

    print(startDate)

    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")
        request.getElement("securities").appendValue(bbg_identifier)
        request.getElement("fields").appendValue("OPEN")
        request.getElement("fields").appendValue("HIGH")
        request.getElement("fields").appendValue("LOW")
        request.getElement("fields").appendValue("PX_LAST")
        request.getElement("fields").appendValue("EQY_WEIGHTED_AVG_PX")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        request.set("startDate", startDate)
        request.set("endDate", endDate)
        request.set("maxDataPoints", 1400) # Don't adjust please :)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)

        # Process received events
        while (True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":

                    histdata = []

                    for fd in msg.getElement("securityData").getElement(
                            "fieldData").values():
                        histdata.append([fd.getElementAsString("date"), \
                                         fd.getElementAsFloat("OPEN"),
                                         fd.getElementAsFloat(
                                             "HIGH"),
                                         fd.getElementAsFloat("LOW"), \
                                         fd.getElementAsFloat("PX_LAST"), \
                                         fd.getElementAsFloat(
                                             "EQY_WEIGHTED_AVG_PX")])

                    histdata = pd.DataFrame(histdata, columns=["Date",
                                                               "Open",
                                                               "High", "Low",
                                                               "Close", "VWAP"])

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completely received, so we could exit
                if 'old_bbg_data' in locals():
                    histdata = pd.concat([histdata, old_bbg_data], axis=0)
                    histdata = histdata.drop_duplicates('Date')
                    histdata = histdata.sort_values('Date')
                    histdata.reset_index(drop=True, inplace=True)

                pd.DataFrame.to_csv(
                    histdata, "bbg_data/" + bbg_identifier + ".csv", index=False
                )

                return histdata
    finally:
        # Stop the session
        session.stop()

__copyright__ = """
Copyright 2012. Bloomberg Finance L.P.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:  The above
copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

####### End of Bloomberg Section -----------------------------------------------
################################################################################
