class Config(object):
    def __init__(self):
        self.model = 'self-attention'
        # self.model = 'textcnn'
        # self.model = 'attention'
        self.checkpoint_dir = "output/self_attention/multi_attention_0802/"
        self.word_vocab_file = 'data/word_vocab.txt'
        self.tag_vocab_file = 'data/tag_vocab.txt'
        self.train_data_files = ['data/data_tech_300.train']
        self.eval_data_files = ['data/data_tech_300.eval']
        self.w2v_path = 'data/w2v_without_entity.vec'
        self.mode = 'train'
        self.num_gpus = 0
        self.num_classes = 8
        self.epoch = 20
        
        # training
        self.learning_rate = 0.01
        self.optimizer = 'adam'
        self.max_gradient_steps = 0
        self.learning_decay = False
        self.start_decay_step = None
        self.decay_factor = 0.98
        self.batch_size = 16
        self.l2_reg_lambda = 0.000001
        self.dropout_keep_prob = 0.6

        # embedding
        self.embedding_size = 150
        self.pretrained_embedding_file = 'data/w2v_without_entity.vec'

        # textcnn
        self.sentence_length = 30
        self.filter_sizes = [1]
        self.num_filters = 100

        # attention
        self.W_dim = 150
        self.hidden_layer_size = 200

        # self-attention
        self.attention_func = None
        self.num_query = 10
        
        


        