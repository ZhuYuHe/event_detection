import numpy as np
def evaluate(model, batch_manager):
    correct_pred = 0
    for batch in batch_manager.next_batch():
        accuracy = model.evaluate(*zip(*batch))[0]
        # print(accuracy, len(batch))
        correct_pred += (int)(accuracy * len(batch))
    return correct_pred / get_dataset_size(batch_manager)

def get_dataset_size(batch_manager):
    res = 0
    for batch in batch_manager.next_batch():
        res += len(batch)
    return res

def evaluate_attention(model, sentences, id2tag):
    correct_num = 0
    tag2id = {tag: idx for idx, tag in id2tag.items()}
    non_event_id = tag2id['__label__非事件']
    for sen in sentences:
        if model.evaluate(sen, non_event_id):
            correct_num += 1

    accuracy = correct_num / len(sentences)
    return accuracy



