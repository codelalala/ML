#!/usr/bin/env python3

from absl import app, flags, logging

import sh

import torch as th
import pytorch_lightning as pl

import nlp
import transformers
import pandas as pd
from datasets import Dataset

flags.DEFINE_boolean('debug',False,'')
flags.DEFINE_integer('epochs',10,'')
flags.DEFINE_integer('batch_size',8,'')
flags.DEFINE_float('lr',2e-5,'')
flags.DEFINE_float('momentum',.5,'')
flags.DEFINE_string('model','bert-base-uncased','')
flags.DEFINE_integer('seq_length',128,'')


FLAGS=flags.FLAGS
sh.rm('-r','-f','logs')
sh.mkdir('logs')

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model=transformers.BertForSequenceClassification.from_pretrained(FLAGS.model,num_labels=2)
        #self.loss=th.nn.CrossEntropyLoss(reduction='none')


    def prepare_data(self):
        tokenizer=transformers.BertTokenizer.from_pretrained(FLAGS.model)
        def _tokenize(x):
            x['input_ids']=tokenizer.encode(
                    x['text'],
                    max_length=FLAGS.seq_length,
                    pad_to_max_length=True)
            return x 
        def _prepare_ds(split):
            #ds=nlp.load_dataset('imdb',split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else "100%"}]')
            #ds=ds.shuffle()
            #import IPython;IPython.embed();exit(1)
            ds=ds.map(_tokenize)
            ds.set_format(type='torch',columns=['input_ids','label'])
            self.ds=ds
            return ds
        #self.train_ds,self.test_ds=map(_prepare_ds,('train','test'))

        df=pd.read_csv('/datasets/IMDB_Dataset.csv')
        #df=df[:1000]
        df=df.rename(columns={"sentiment":"label","review":"text"})
        df['label']=df['label'].replace('positive',1)
        df['label']=df['label'].replace('negative',0)
        ds=Dataset.from_pandas(df)
        ds=ds.map(_tokenize)
        ds.set_format(type='torch',columns=['input_ids','label'])
        ds=ds.train_test_split(test_size=0.1,shuffle=True)
        self.train_ds,self.test_ds=ds['train'],ds['test']
        #import IPython;IPython.embed();exit(1)
        return ds

    def forward(self,input_ids):
        pass
        #mask=(input_ids!=0).float()
        #logits=self.model(input_ids,mask).logits
        #return logits
        #import IPython;IPython.embed();exit(1)

    def training_step(self,batch,batch_idx):
        #logits=self.forward(batch['input_ids'])
        #loss=self.loss(logits,batch['label']).mean()
        input_ids=batch['input_ids']
        labels=batch['label']
        output=self.model(input_ids,labels=labels)
        loss=output.loss
        #import IPython;IPython.embed();exit(1)
        self.log('train_loss',loss)
        return {'loss':loss}
        

    def validation_step(self,batch,batch_idx):
        #import IPython;IPython.embed();exit(1)
        #logits=self.forward(batch['input_ids'])
        #loss=self.loss(logits,batch['label'])
        input_ids=batch['input_ids']
        labels=batch['label']
        output=self.model(input_ids,labels=labels)
        loss=output.loss
        acc=(output.logits.argmax(-1)==batch['label']).float()
        return {'loss':loss,'acc':acc}

    def validation_epoch_end(self,outputs):
        loss=th.stack([o['loss'] for o in outputs],0).mean()
        acc=th.cat([o['acc'] for o in outputs],0).mean()
        out={'val_loss':loss,'val_acc':acc}
        self.log('val_loss',loss)
        self.log('val_acc',acc)
        #import IPython;IPython.embed();exit(1)
        return out

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True
                )
    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=False
                )


    def configure_optimizers(self):
        WEIGHT_DECAY = 0.01
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': WEIGHT_DECAY},
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
            ]
        return th.optim.AdamW(
                #self.parameters(),
                optimizer_grouped_parameters,
                lr=FLAGS.lr,
                )

def main(_):
    model=IMDBSentimentClassifier()
    trainer=pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if th.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=pl.loggers.TensorBoardLogger('logs/',name='imdb',
                version=0,default_hp_metric=False),
            #log_every_n_steps=1,
            checkpoint_callback=False,
            )
    trainer.fit(model)


if __name__=='__main__':
    app.run(main)

