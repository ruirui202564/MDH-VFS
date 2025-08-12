def config():
    params_list = {'referNum': [4,5],
                   'referValue': ['center'],
                   'warningNum': [10, 15],#N
                   'updateThres': [80, 100], #s
                   'window': [400, 500]}

    return params_list

def create_param_combination(model_params):
    param_combination = []
    for a in model_params['referNum']:
        for b in model_params['referValue']:
            for c in model_params['warningNum']:
                for d in model_params['updateThres']:
                    for e in model_params['window']:
                        param_combination.append({'referNum': a, 'referValue': b, 'warningNum': c,
                                                  'updateThres': d, 'window':e})
    return param_combination