import os
import json
import logging
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from models import KGEModel, ModE, HAKE

from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='runs.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('-mw', '--modulus_weight', default=1.0, type=float)
    parser.add_argument('-pw', '--phase_weight', default=0.5, type=float)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--no_decay', action='store_true', help='Learning rate do not decay')
    parser.add_argument('--use_mean_aggregator', help='aggregator information of past timestamps', type=bool, default=False)
    parser.add_argument('--prev_num_steps', default=5, type=int, help='prev time steps to use')
    parser.add_argument('--use_bert', help='Use bert to initialize the embeddings of entity/relation', type=bool, default=False)
    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as f:
        args_dict = json.load(f)

    args.model = args_dict['model']
    args.data_path = args_dict['data_path']
    args.hidden_dim = args_dict['hidden_dim']
    args.test_batch_size = args_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    args_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )

def convert_word2emb(ele2id):
    word_emb = torch.zeros(len(ele2id), 768)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertModel.from_pretrained('bert-base-cased').cuda()
    model.eval()
    for name, id in ele2id.items():
        print("Convert word: (", name, ") to Bert embedding")
        text = "[CLS] " + name + " [SEP]"
        tokens=tokenizer.tokenize(text)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # input_mask = [1] * len(input_ids)

        # padding = [0] * (self.max_seq_length - len(input_ids))
        # input_ids += padding
        # input_mask += padding
        # segment_ids += padding

        tokens_tensor = torch.tensor([input_ids], dtype=torch.long).cuda()
        segments_tensor = torch.tensor([segment_ids], dtype=torch.long).cuda()
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensor)

            batch_i = 0
            token_embeddings = []

            for token_i in range(len(input_ids)):
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    vec = encoded_layers[layer_i][batch_i][token_i]
                    hidden_layers.append(vec)
                token_embeddings.append(hidden_layers)
            sentence_embedding = torch.mean(encoded_layers[11], 1)
            word_emb[id]=sentence_embedding

    return word_emb



def save_model_time(model, t, args):
    torch.save(model, os.path.join(args.save_path, "model_time_" + str(t)+".chkpnt"))

def load_model_time(t, args):
    return torch.load(os.path.join(args.save_path, "model_time_" + str(t)+".chkpnt"))

def mean_aggregator(model, prev_model_list):
    for m in prev_model_list:
        model.entity_embedding.data = (model.entity_embedding.data + m.entity_embedding.data)
        model.relation_embedding.data = (model.relation_embedding.data + m.relation_embedding.data)
        model.phase_weight.data = (model.phase_weight.data + m.phase_weight.data)
        model.modulus_weight.data = (model.modulus_weight.data + m.modulus_weight.data)
    model.entity_embedding.data = (model.entity_embedding.data/(len(prev_model_list)+1))
    model.relation_embedding.data = (model.relation_embedding.data/(len(prev_model_list)+1))
    model.phase_weight.data = (model.phase_weight.data/(len(prev_model_list)+1))
    model.modulus_weight.data = (model.modulus_weight.data/(len(prev_model_list)+1))

    #print(model.state_dict())
    #assert False
    return model

def adpative_mean_aggregator(model, prev_model_list):
    print("Here in adapative mean aggregator, length of previous_model_list: ", len(prev_model_list))
    model.entity_embedding.data = model.entity_embedding.data - model.entity_embedding.data
    model.relation_embedding.data = model.relation_embedding.data - model.relation_embedding.data
    model.phase_weight.data = model.phase_weight.data - model.phase_weight.data
    model.modulus_weight.data = model.modulus_weight.data - model.modulus_weight.data
    for i, m in enumerate(prev_model_list):
        model.entity_embedding.data = (model.entity_embedding.data + (i+1)*m.entity_embedding.data)
        model.relation_embedding.data = (model.relation_embedding.data + (i+1)*m.relation_embedding.data)
        model.phase_weight.data = (model.phase_weight.data + (i+1)*m.phase_weight.data)
        model.modulus_weight.data = (model.modulus_weight.data + (i+1)*m.modulus_weight.data)
    sum = 0
    for i in range(len(prev_model_list)+1):
        sum += i
    model.entity_embedding.data = (model.entity_embedding.data/sum)
    model.relation_embedding.data = (model.relation_embedding.data/sum)
    model.phase_weight.data = (model.phase_weight.data/sum)
    model.modulus_weight.data = (model.modulus_weight.data/sum)

    #print(model.state_dict())
    #assert False
    return model

def get_elements_TKG(TKG):
    entities, relations = set(), set()
    for triple in TKG:
        entities.add(triple[0])
        entities.add(triple[2])
        relations.add(triple[1])
    return entities

def get_aggregator_coeff(elements_TKG, entities_now):
    coefficients = []
    for entities_t in elements_TKG:
        coefficients.append(-len(entities_t & entities_now))
    coefficients = np.exp(coefficients)
    return coefficients/np.sum(coefficients)

def weighted_mean(model, prev_model_list, coefficients):
    print("Here in weighted mean aggregator, length of previous_model_list: ", len(prev_model_list), "previous_coefficients _list: ", coefficients)
    model.entity_embedding.data = model.entity_embedding.data - model.entity_embedding.data
    model.relation_embedding.data = model.relation_embedding.data - model.relation_embedding.data
    model.phase_weight.data = model.phase_weight.data - model.phase_weight.data
    model.modulus_weight.data = model.modulus_weight.data - model.modulus_weight.data

    for coeff, m in zip(coefficients, prev_model_list):
        model.entity_embedding.data = model.entity_embedding.data + coeff * m.entity_embedding.data
        model.relation_embedding.data = model.relation_embedding.data + coeff * m.relation_embedding.data
        model.phase_weight.data = model.phase_weight.data + coeff * m.phase_weight.data
        model.modulus_weight.data = model.modulus_weight.data + coeff * m.modulus_weight.data
    return model

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    if "YAGO" in args.data_path:
        data_reader = DataReader(args.data_path, "YAGO")
    elif "WIKI" in args.data_path:
        data_reader = DataReader(args.data_path, "WIKI")
    else:
        data_reader = DataReader(args.data_path)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)

    logging.info('Model: {}'.format(args.model))
    logging.info('Data Path: {}'.format(args.data_path))
    logging.info('Num Entity: {}'.format(num_entity))
    logging.info('Num Relation: {}'.format(num_relation))

    logging.info('Num Train: {}'.format(len(data_reader.static_train_data)))
    logging.info('Num Valid: {}'.format(len(data_reader.static_valid_data)))
    logging.info('Num Test: {}'.format(len(data_reader.static_test_data)))

    if args.model == 'ModE':
        kge_model = ModE(num_entity, num_relation, args.hidden_dim, args.gamma)
    elif args.model == 'HAKE':
        if args.use_bert == True:
            entity_word_emb = convert_word2emb(data_reader.entity_dict)
            relation_word_emb = convert_word2emb(data_reader.relation_dict)
            kge_model = HAKE(num_entity, num_relation, args.hidden_dim, args.gamma, args.modulus_weight, args.phase_weight, entity_word_emb, relation_word_emb)
        else:
            kge_model = HAKE(num_entity, num_relation, args.hidden_dim, args.gamma, args.modulus_weight, args.phase_weight)

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.HEAD_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.TAIL_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )

        warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    if not args.do_test:
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.do_train:
        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                if not args.no_decay:
                    current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                #save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, data_reader, ModeType.VALID, args)
                log_metrics('Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        #save_model(kge_model, optimizer, save_variable_list, args)
    '''
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, data_reader, ModeType.VALID, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, data_reader, ModeType.TEST, args)
        log_metrics('Test', step, metrics)
    '''





    test_result = []
    head_result = []
    tail_result = []
    test_num = 0
    time_list = list(data_reader.TKG_train.keys())
    time_list.sort()

    mrr_times=[]
    loss_times=[]
    model_prev_list, elements_prev_list = [], []
    for i in time_list:
        if i not in data_reader.TKG_test.keys() or i not in data_reader.TKG_valid.keys():
            continue

        if args.use_mean_aggregator:
            if len(model_prev_list) < args.prev_num_steps:
                if i > 0:
                    model_prev_list.append(load_model_time(i-1, args))
                    elements_prev_list.append(get_elements_TKG(data_reader.TKG_train[i-1]))
            else:
                del model_prev_list[0]
                model_prev_list.append(load_model_time(i-1, args))

                del elements_prev_list[0]
                elements_prev_list.append(get_elements_TKG(data_reader.TKG_train[i-1]))
            #if i >= args.prev_num_steps:
            if i > 1:
                entities_now = get_elements_TKG(data_reader.TKG_train[i])
                #kge_model=adpative_mean_aggregator(kge_model, model_prev_list)
                coefficients = get_aggregator_coeff(elements_prev_list, entities_now)
                kge_model = weighted_mean(kge_model, model_prev_list, coefficients)

        logging.info("Time: "+str(i))
        train_dataloader_head = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.HEAD_BATCH, True, i),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.TAIL_BATCH, True, i),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        training_logs = []

        # Training Loop
        loss_i=[]
        for step in range(init_step, args.max_steps//20):
        #for step in range(init_step, 0):
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)

            training_logs.append(log)
            loss_i.append(log["loss"])

            if step >= warm_up_steps:
                if not args.no_decay:
                    current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0 and step > 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                #save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, data_reader, ModeType.VALID, args, True, i)
                log_metrics('Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }

        if args.do_valid:
            logging.info('Evaluating on Valid Dataset...')
            metrics = kge_model.test_step(kge_model, data_reader, ModeType.VALID, args, True, i)
            log_metrics('Valid', step, metrics)

        if args.do_test:
            logging.info('Evaluating on Test Dataset...')
            if "YAGO" in args.data_path or "WIKI" in args.data_path:
                metrics, head_metrics, tail_metrics = kge_model.test_step(kge_model, data_reader, ModeType.TEST, args, True, i, True)
            else:
                metrics = kge_model.test_step(kge_model, data_reader, ModeType.TEST, args, True, i)
            log_metrics('Test', step, metrics)

            metric_names = list(metrics.keys())
            metric_names.sort()
            test_result.append([metrics[m]*len(data_reader.TKG_test[i]) for m in metric_names])
            test_num+=len(data_reader.TKG_test[i])

            if "YAGO" in args.data_path or "WIKI" in args.data_path:
                head_result.append([head_metrics[m]*len(data_reader.TKG_test[i]) for m in metric_names])
                tail_result.append([tail_metrics[m]*len(data_reader.TKG_test[i]) for m in metric_names])

        if args.use_mean_aggregator:
            save_model_time(kge_model, i, args)

        mrr_times.append(metrics['MRR'])
        loss_i=np.mean(loss_i)
        loss_times.append(loss_i)

    print(metric_names)
    print("Dynamic_test_result: ", np.sum(test_result, axis=0)/test_num)
    print("Head_result: ", np.sum(head_result, axis=0)/test_num)
    print("Tail_result: ", np.sum(tail_result, axis=0)/test_num)
    print(mrr_times)
    print(loss_times)
    np.save(
        os.path.join(args.save_path, 'mrr_times'),
        np.array(mrr_times)
    )
    np.save(
        os.path.join(args.save_path, 'loss_times'),
        np.array(loss_times)
    )



if __name__ == '__main__':
    main(parse_args())
