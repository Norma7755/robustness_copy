from models.SSM import SSM, SSM_Individual_Head, Mega, S5_SSM, S6_SSM, Transformer

def build_model(args, model_name):
    if model_name == 'SSM':
        if args.use_inject:
            model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, \
                use_inject=True,inject_method=args.inject_method)
        else:
            model = SSM(d_input=1, d_model=128, n_layers=args.num_layers)
    elif model_name == 'DSS':
        if args.use_inject:
            model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, mode = 'diag', \
                use_inject=True,inject_method=args.inject_method)
        else:
            model = SSM(d_input=1, d_model=128, n_layers=args.num_layers, mode = 'diag')
    elif model_name == 'S5':
        model = S5_SSM(d_input=1, d_model=128, n_layers=args.num_layers)
    elif model_name == 'Mega':
        model = Mega(d_input=1, d_model=128, n_layers=args.num_layers, seq_len=28*28) 
    elif model_name == 'S6':
        model = S6_SSM(d_input=1, d_model=128, n_layers=args.num_layers)     
    elif model_name == 'SSM_ind_head':
        model = SSM_Individual_Head(d_input=1)
    elif model_name == 'Transformer':
        model = Transformer(d_input=1, n_layers=args.num_layers, patch_size=2)
    return model