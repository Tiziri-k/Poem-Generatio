from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import  DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import os


class GPT2_poem:
    def __init__(self, data, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_type)
        self.text = []  
        for row in data['content']:
            self.text.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))
        if truncate:
            self.text = self.text[:20000]
        self.text_count = len(self.text)

    def pack_tensor(self, new_tensor, packed_tensor, max_seq_len):
        if packed_tensor is None:
            return new_tensor, True, None
        if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
            return packed_tensor, False, new_tensor
        else:
            packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
            return packed_tensor, True, None

        
    def train(self, batch_size=16, epochs=5, lr=2e-5, max_seq_len=400, warmup_steps=200, gpt2_type="gpt2", output_dir="/modeles", output_prefix="wreckgar", test_mode=False, save_model_on_epoch=False):
        acc_steps = 100
        device = torch.device("cuda")
        self.model = self.model.cuda()
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
        )

        train_dataloader = DataLoader(self.text, batch_size=1, shuffle=True)
        
        loss = 0
        accumulating_batch_count = 0
        input_tensor = None

        for epoch in range(epochs):
            print(f"Training epoch {epoch}")
            print(loss)
            for idx, entry in tqdm(enumerate(train_dataloader)):
                (input_tensor, carry_on, remainder) = self.pack_tensor(entry, input_tensor, max_seq_len)

                if carry_on and idx != len(train_dataloader) - 1:
                    continue

                input_tensor = input_tensor.to(device)
                outputs = self.model(input_tensor, labels=input_tensor)
                loss = outputs[0]
                loss.backward()

                if (accumulating_batch_count % batch_size) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()

                accumulating_batch_count += 1
                input_tensor = None

            if save_model_on_epoch:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
                )
        return self.model
    
    def generate_poem(self, prompt, max_length=200):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        device = self.model.device
        input_ids = input_ids.to(device)

        attention_mask = torch.ones(input_ids.shape, device=device)

        pad_token_id = self.tokenizer.eos_token_id

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            attention_mask=attention_mask,  
            pad_token_id=pad_token_id  
        )

        generated_poem = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_poem


#         if tokenizer.pad_token is None:
#             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#             model.resize_token_embeddings(len(tokenizer))  

# def generate_poem(keywords, num_poems=1, temperature=0.8):
#     if not keywords.strip():
#         raise ValueError("Keywords input cannot be empty.")

#     # Tokenize the input and create an attention mask
#     inputs = tokenizer.encode(keywords, return_tensors='pt', padding=True, truncation=True)
#     attention_mask = tokenizer.encode_plus(
#         keywords,
#         return_tensors='pt',
#         padding=True,
#         truncation=True
#     )["attention_mask"]

#     combined_poem = ""
#     for i in range(num_poems):
#         outputs = model.generate(
#             inputs,
#             max_length=150,
#             do_sample=True,
#             top_p=0.92,
#             temperature=temperature,
#             num_return_sequences=1,
#             pad_token_id=tokenizer.pad_token_id,
#             attention_mask=attention_mask
#         )
#         poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         combined_poem += poem + "\n\n"

#     return combined_poem
