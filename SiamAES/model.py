import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, TokenClassifierOutput

class BaselineModel(torch.nn.Module):
  def __init__(self,num_classes): 
    super(BaselineModel,self).__init__() 
    self.num_labels = num_classes


    self.dropout = nn.Dropout(0.2) 

    # Configure DistilBERT's initialization
    config = DistilBertConfig(output_hidden_states=True, dropout=0.2)

    self.model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
    for name, param in list(self.model.named_parameters())[:-2]: 
        param.requires_grad = False
    self.classifier = nn.Linear(768, num_classes) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None, labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
    embed = sequence_output[:,0,:].view(-1,768)
    logits = self.classifier(embed) # calculate losses
    
    target = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).double()
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), target)

    return (TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions), embed, target)
  
class SiameseNetworkLSTM(nn.Module):
    def __init__(self, num_labels):
        super(SiameseNetworkLSTM, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.3) 
        self.LSTM = nn.LSTM(768, 768, bias=True, bidirectional=True)
        self.LSTM.train()
        # Configure DistilBERT's initialization
        config = DistilBertConfig(output_hidden_states=True, dropout=0.1)

        self.model = model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        for name, param in list(self.model.named_parameters())[:-2]: 
            param.requires_grad = False
        
        self.classifier = nn.Linear(768, num_labels) # load and initialize weights
        torch.nn.init.xavier_uniform(self.classifier.weight)

    def forward_one(self, input_ids=None, attention_mask=None, labels=None):
        token_embeds = self.model.get_input_embeddings().weight[input_ids].detach().clone()
        token_embeds.requires_grad = True
        outputs = self.model(inputs_embeds=token_embeds, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        # print(sequence_output.size())
        encoder_input = sequence_output[0,:,:].squeeze(0).view(-1,768)
        # print("LSTM_input:", encoder_input.size())
        LSTM_values = self.LSTM(encoder_input)
        # embedding corresponding to final hidden state
        embed = LSTM_values[1][0]
        # average beginning and end vectors
        embed = (torch.add(embed[0,:], embed[1,:]) / 2).unsqueeze(0)
        # print("embedding size:", embed.size())
        return LSTM_values[0], embed, token_embeds


    def forward(self, x1, x2, x3=None):

        outputs_x1, out_x1, token_embeds_x1 = self.forward_one(**x1)
        logits_x1 = self.classifier(out_x1)
        target_x1 = torch.nn.functional.one_hot(x1['labels'], num_classes=self.num_labels).double()
        args_x1 = (logits_x1, target_x1, outputs_x1, token_embeds_x1)

        if(x2 is not None):
          outputs_x2, out_x2, token_embeds_x2 = self.forward_one(**x2)
          logits_x2 = self.classifier(out_x2)
          target_x2 = torch.nn.functional.one_hot(x2['labels'], num_classes=self.num_labels).double()
          args_x2 = (logits_x2, target_x2, outputs_x2, token_embeds_x2)
          return out_x1, out_x2, args_x1, args_x2
        else:
          # grad = torch.autograd.grad(logits_x1[0][0], token_embeds_x1);
          args_x1 = (logits_x1, target_x1, outputs_x1)
          return out_x1, args_x1
        
class SiameseNetwork(nn.Module):

    def __init__(self, num_labels):
        super(SiameseNetwork, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1) 

        # Configure DistilBERT's initialization
        config = DistilBertConfig(output_hidden_states=True, dropout=0.1)

        self.model = model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        for name, param in list(self.model.named_parameters())[:-2]: 
            param.requires_grad = False
        
        self.classifier = nn.Linear(768, num_labels) # load and initialize weights
        torch.nn.init.xavier_uniform(self.classifier.weight)

    def forward_one(self, input_ids=None, attention_mask=None, labels=None):
        token_embeds = self.model.get_input_embeddings().weight[input_ids].detach().clone()
        token_embeds.requires_grad = True
        outputs = self.model(inputs_embeds=token_embeds, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

        embed = sequence_output[:, 0, :].view(-1, 768)#torch.sum(sequence_output,1).view(-1,768)
        return outputs, embed, token_embeds


    def forward(self, x1, x2, x3):

        outputs_x1, out_x1, token_embeds_x1 = self.forward_one(**x1)
        logits_x1 = self.classifier(out_x1)
        target_x1 = torch.nn.functional.one_hot(x1['labels'], num_classes=self.num_labels).double()
        args_x1 = (logits_x1, target_x1, outputs_x1, token_embeds_x1)

        if(x2 is not None):
          outputs_x2, out_x2, token_embeds_x2 = self.forward_one(**x2)
          logits_x2 = self.classifier(out_x2)
          target_x2 = torch.nn.functional.one_hot(x2['labels'], num_classes=self.num_labels).double()
          args_x2 = (logits_x2, target_x2, outputs_x2, token_embeds_x2)
          if(x3 is None):
            return out_x1, out_x2, args_x1, args_x2
          else:
            outputs_x3, out_x3, token_embeds_x3 = self.forward_one(**x2)
            logits_x3 = self.classifier(out_x3)
            target_x3 = torch.nn.functional.one_hot(x3['labels'], num_classes=self.num_labels).double()
            args_x3 = (logits_x3, target_x3, outputs_x3, token_embeds_x3)
            return out_x1, out_x2, out_x3, args_x1, args_x2, args_x3
        else:
          y = logits_x1[0]
          grad = torch.autograd.grad(outputs=y, inputs=token_embeds_x1, grad_outputs=torch.ones_like(y))
          args_x1 = (logits_x1, target_x1, outputs_x1, grad)
          return out_x1, args_x1
                                
                              
class CrossEntropyLoss(torch.nn.Module):
  def __init__(self, num_labels):
    super(CrossEntropyLoss, self).__init__()  # pre 3.3 syntax
    self.num_labels = num_labels
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, x):
    loss_fct = nn.CrossEntropyLoss()
    logits = x[0]; target = x[1]; outputs = x[2]
    #print(self.softmax(logits), target)
    soft = self.softmax(logits)
    loss = loss_fct(logits, target)
    entropy =  soft@torch.log(soft).T
    gamma = 0.3
    return gamma*entropy + loss