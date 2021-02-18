# -*- coding: utf-8 -*-
import sys
import get_dataset as data
import multi_model
from keras.utils import to_categorical
import time
import test_scores as score


def DeepW(file_name,kmer,seq_kernel,struc_kernel):
    model_path = './model'
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    # get training data and validation data
    print('get training data')
    train_file = './datasets/clip/' + file_name + '/30000/training_sample_0'
    train_data = data.dataset(train_file,kmer,seq_kernel,struc_kernel)
    train_label = train_data[1]
    train_indice, train_y, validation_indice, validation_y = data.split_training_validation(train_label)

    train_seq = train_data[0][0][train_indice]
    val_seq = train_data[0][0][validation_indice]
    train_struc = train_data[0][1][train_indice]
    val_struc = train_data[0][1][validation_indice]

    train_y = to_categorical(train_y, 2)
    val_y = to_categorical(validation_y, 2)

    # get test data
    print('get test data')
    test_file = './datasets/clip/' + file_name + '/30000/test_sample_0'
    test_data = data.dataset(test_file,kmer,seq_kernel,struc_kernel)
    test_seq = test_data[0][0]
    print(test_seq.shape)
    test_struc = test_data[0][1]
    print(test_struc.shape)
    test_label = test_data[1]
    test_y = to_categorical(test_label)
    # 3mer model
    if kmer == 1:
        print('1mer model training...')
        dw_model = multi_model.merged_BLSTM_CNN_1mer(train_seq.shape[1],train_struc.shape[1],seq_kernel, struc_kernel)
    elif kmer == 2:
        print('2mer model training...')
        dw_model = multi_model.merged_BLSTM_CNN_2mer(train_seq.shape[1],train_struc.shape[1],seq_kernel, struc_kernel)
    elif kmer == 3:
        print('3mer model training...')
        dw_model =multi_model.merged_BLSTM_CNN_3mer(train_seq.shape[1],train_struc.shape[1],seq_kernel, struc_kernel)
    dw_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    dw_model.fit(x=[train_seq, train_struc], y=train_y, batch_size=50, epochs=30, validation_data=([val_seq,val_struc], val_y), verbose=1,shuffle=True)
    # save 2mer model
    #dw_model.save(model_path+'/2mer_model/'+file_name[:10]+'.h5')
    prob = dw_model.predict([test_seq,test_struc])
    y_pred_cnn = score.transfer_label_from_prob(prob[:,-1])
    y_prob = prob[:,-1]
    acc, balanced_acc, AUC = score.calculate_performace(len(y_pred_cnn), y_pred_cnn, y_prob, test_y[:,-1])
    print(str(kmer)+'mer_model:\t'+'ACC:'+str(acc)+'\t'+'balanced_acc:'+str(balanced_acc)+'\t'+'AUC:'+str(AUC)+'\n')
    with open('./results/'+str(kmer)+'mer_model_kernel_'+str(seq_kernel)+'_'+str(struc_kernel)+'_BLSTM_compared.txt','a') as fw:
        fw.write(file_name+'_'+str(seq_kernel)+'_'+str(struc_kernel)+'\tACC:\t'+str(acc)+'\t'+'balanced_acc:\t'+str(balanced_acc)+'\t'+'AUC:\t'+str(AUC)+'\n')


if __name__ == '__main__':
    kmer = 2
    seq_kernel = 4
    struc_kernel = 4
    dataset = '18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome'
    DeepW(dataset,kmer,seq_kernel,struc_kernel)



