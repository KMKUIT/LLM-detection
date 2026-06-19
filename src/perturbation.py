from __future__ import annotations

def generate_perturbations(sentence:str,num_perturbations:int,model,tokenizer,device):

    model.to(device)

    text =  "paraphrase: " + sentence + " </s>"

    max_len = 256

    encoding = tokenizer(text,padding="max_length",truncation=True,max_length=max_len,return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    print("Generating perturbations...")
    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    # Have to read about these decodings
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=max_len,
        top_k=220,
        top_p=1,
        early_stopping=True,
        num_return_sequences=num_perturbations
    )
    print("DONE\n")

    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        final_outputs.append(sent)
    
    return final_outputs
