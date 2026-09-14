from sacred import Experiment
from exctransformer.utils.configchecks import _loss_names

ex = Experiment("exctransformer", interactive=False, save_git_info=False)

@ex.config
def config():

    # model
    seed = 0
    exp_name = 'exctransformer'

    # Graphsage Model configuration
    max_crystal_atoms = 500
    num_layers = 4
    hidden_dims = 128
    graph_attn_heads = 6                  # Consider ensuring that hidden_dims * num_heads = out_dim
    out_dim = 768                                    # This allows richer embeddings without needing projection
    

    # Transformer Model configuration
    hid_dim = 768                # 768 for BERT model
    transformer_num_layers = 12    # 12 for BERT model
    num_heads = 12                 # 12 for BERT model   
    mlp_ratio = 4.0
    drop_rate= 0.1

    # Downstream Model configuration
    num_targets = 1               # For a regression tasks
    num_classes = 0            # For a classification tasks

    # Training configuration
    max_epochs = 30
    loss_names = _loss_names({"regression": 1})


    # Other configuration parameters
    drop_rate = 0.1
    mpp_ratio = 0.15

    # Optimizer Setting
    optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 1e-4
    weight_decay = 1e-2
    decay_power = (
        1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    )
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.05  
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream head


    # PL Trainer variables
    resume_from = None
    val_check_interval = 1.0
    test_only = False
    load_path = None # Users can select default models by specifying either "exctransformer", "moftransformer", or "pmtransformer"
    no_labels = False # If True, ignores labels/targets during prediction mode inference
    load_pretained_weights = None # Users can select default models by specifying either "exctransformer", "moftransformer", or "pmtransformer"
   

    # Environment variables
    log_dir = "logs/"
    load_version = None
    batch_size_grad = 32  # desired batch size; for gradient accumulation
    per_gpu_batchsize = 8  # user should define this manually with per_gpu_batch_size
    num_nodes = 1
    accelerator = "auto"
    device = "auto"
    precision = 16
    set_float32_matmul_precision = None
    log_every_n_steps = 10

    # normalization additional features
    mean_a= None
    std_a = None
    graph_only_mode = False  # If True, ignore additional features even if present.

    # normalization targets
    mean = None
    std = None

    # DataModule configuration
    dataset_path = ''
    batch_size = 32
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    sub_data =None
    train_size = None
    val_size = None
    test_size = None
    zipped = False
    lazy = True
    transform = None
    num_workers = 0

    # data_build configuration
    data_dir = "data_test"               # Path to your data directory
    cif_id_prop = "id_prop_reg_h2"     # CSV file (without .csv extension) with CIF IDs and properties
    cif_path_z = "hMof_cifs.zip"        # Directory containing CIF files
    additional_cols = None               # Additional columns to include in the cif_id_prop CSV file
    target_cols = None               # Target columns to include in the cif_id_prop CSV file
    dataset_size = 1000             # Number of CIF files to process
    random_seed = 123
    radius = 8.0
    dmin = 0.0
    dmax = 8.0                    # Also serves as the cutoff radius for neighbor search
    step = 0.2
    output_h5 = "graph_data"     # Output HDF5 file for storing preprocessed graphs
    zip = True                     # If True, load CIF files from a zip archive
    zip_output_path = None
    zip_compressed_data = False
    resume_load = False
    predict_set=False
    

@ex.named_config
def arc_test(): 
    dataset_path = 'graph_data'
    log_dir = "logs/"
    load_version = "version_0"
    seed = 0
    test_only = True
    
@ex.named_config
def knust_run():
    #log_dir = "logs/"
    #load_version = "version_0"
    #seed = 0
    #max_epochs = 100
    #train_size = 100000
    #val_size = 20000
    #test_size = 5600
    #load_pretained_weights = "moftransformer"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    num_targets = 3
    set_float32_matmul_precision = 'medium' 

@ex.named_config
def knust_test():
    dataset_path = 'graph_data_125629'
    log_dir = "logs/"
    load_version = "version_0"
    seed = 0
    max_epochs = 10
    train_size = 7
    val_size = 2
    test_size = 1
    load_pretained_weights = "moftransformer"
    num_targets = 3
    sub_data =10 
