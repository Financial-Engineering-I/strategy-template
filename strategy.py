from sklearn import linear_model
import numpy as np

def strategy(
        trade_dt_var, response_var, features_and_responses, trading_date, N, n
):
    training_indices = features_and_responses[trade_dt_var] < trading_date
    training_X = features_and_responses[training_indices].tail(N)[
        ['a', 'b', 'R2', 'ivv_vol']
    ]
    training_Y = features_and_responses[training_indices].tail(N)[response_var]

    # Need at least two 1's to train a model
    if sum(training_Y) < 2:
        return 0

    if sum(training_Y) < n:
        logisticRegr = linear_model.LogisticRegression()
        logisticRegr.fit(np.float64(training_X), np.float64(training_Y))
        trade_decision = logisticRegr.predict(
            np.float64(
                features_and_responses[["a", "b", "R2", "ivv_vol"]][
                    features_and_responses['Date'] == trading_date
                    ]
            )
        ).item()
    else:  # If EVERYTHING is a 1, then just go ahead and implement again.
        trade_decision = 1

    return trade_decision
