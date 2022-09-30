import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange

from datasets import my_collate
torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_input_from_batch(batch):
    inputs = { 'token_ids':batch[0],
               'event_ids':batch[1],
               'token_adj':batch[2],
               'phrase_adj':batch[3],
               'structure_adj':batch[4],
               'event_adj':batch[5],
               'token2phrase':batch[6],
               'phrase2structure':batch[7],
               'structure2event':batch[8]
                }
    labels = batch[9]
    return inputs, labels

def get_collate_fn():
    return my_collate

def train(args,model,train_dataset,train_labels_weight,test_dataset,test_labels_weight):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn()
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=100,verbose=True,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epoch = 0
    f = open('././output/result.txt', 'w', encoding='utf-8-sig')
    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs, labels = get_input_from_batch(batch)
            logit = model(**inputs)
            loss = F.cross_entropy(logit, labels,weight=train_labels_weight.to(args.device))
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args,test_dataset,model,test_labels_weight,f)
                    # scheduler.step(results['macro_f1'])  # Update learning rate schedule
                    all_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("  train_loss: %s", str((tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

        epoch += 1
        tb_writer.add_scalar('train_epoch_loss',(tr_loss - logging_loss) / args.logging_steps, epoch)

    tb_writer.close()
    # torch.save(model, './output/train.model')
    # evaluate(args, test_dataset,model)
    return global_step, tr_loss/global_step, all_eval_results

def evaluate(args, eval_dataset, model,test_labels_weight,writer):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn()
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(batch)
            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels,weight=test_labels_weight.to(args.device))
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
        torch.cuda.empty_cache()

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    logger.info('***** Eval results *****')
    logger.info("  eval loss: %s", str(eval_loss))
    for key in result.keys():
        logger.info("  %s = %s", key, str(result[key]))

    target_names = ['Causality','Constituent','Contrast','Degree','Inference','Parataxis','Phenomenon-Instance',
                 'Supplement','Temporality','Other']
    class_result = classification_report(y_true=out_label_ids, y_pred=preds, target_names=target_names, digits=4)
    writer.write(class_result)
    writer.write('macro:' + str(result['macro_f1']))
    return results, eval_loss

def compute_metrics(preds, labels):
    pre = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "pre":pre,
        "recall":recall,
        "macro_f1": macro_f1
    }
