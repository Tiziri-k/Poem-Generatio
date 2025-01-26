from transformers import pipeline 


def generate_poemNeo(prompt):
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")     
    
    res = generator(prompt, max_length=100, do_sample=True, temperature=0.9) 
    

    return res[0]['generated_text']

