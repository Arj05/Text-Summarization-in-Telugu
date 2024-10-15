#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Section 1: Install Required Libraries
get_ipython().system('pip install transformers datasets')

# Section 2: Import Libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

# Section 3: Load and Preprocess the Dataset
data_path = r'C:\Users\arjun\OneDrive\Desktop\telugu_ilsum_2024_train.csv'  # Update the path as needed
df = pd.read_csv(data_path, encoding='ISO-8859-1')  # Specify encoding to avoid UnicodeDecodeError
print(df.head())

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into training and validation sets
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
dataset_train = train_test_split['train']
dataset_valid = train_test_split['test']

# Section 4: Initialize the Tokenizer and Model
MODEL = 'google/mt5-small'  # You can choose a different pre-trained model if desired
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)  # Ensure this is executed to define `model`

# Section 5: Data Preprocessing Function
def preprocess_function(examples, tokenizer):
    inputs = [f"summarize: {article}" for article in examples['Heading']]
    model_inputs = tokenizer(
        inputs,
        max_length=512,  # Adjust as necessary
        truncation=True,
        padding='max_length'
    )

    targets = [summary for summary in examples['Summary']]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=150,  # Adjust as necessary
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Section 6: Apply the Map Function
tokenized_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=4,  # You can adjust the number of processes
    fn_kwargs={'tokenizer': tokenizer}  # Pass tokenizer as an argument
)

tokenized_valid = dataset_valid.map(
    preprocess_function,
    batched=True,
    num_proc=4,  # You can adjust the number of processes
    fn_kwargs={'tokenizer': tokenizer}  # Pass tokenizer as an argument
)

# Section 7: Set Up Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  # Use `eval_strategy` instead of `evaluation_strategy`
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,  # Ensure model is defined in the previous section
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
)

# Section 8: Train the Model
trainer.train()

# Section 9: Evaluate the Model
results = trainer.evaluate()
print(results)

# Section 10: Generate Summaries
def generate_summary(text):
    input_ids = tokenizer(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True).input_ids
    summary_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example of generating a summary
sample_text = """
సినిమా అనేది మానవ జీవితంలో ఒక ప్రత్యేకమైన స్థానం కలిగి ఉంది. ఇది కేవలం వినోదం కాదు, ఇది భావనలను, భావోద్వేగాలను, మరియు అనుభూతులను వ్యక్తం చేసే ఒక సాధనంగా మారింది. తెలుగు సినిమా పరిశ్రమ, అందులోని సృజనాత్మకత, ప్రతిభ, మరియు ప్రతిభావంతమైన దర్శకులు, నటులు అందరి కలయికతో అత్యంత ప్రాచుర్యం పొందింది.

తెలుగు సినిమాలు ఆర్థికంగా, సామాజికంగా, మరియు సాంస్కృతికంగా అనేక మార్పులను చూపించాయి. 1950ల నాటి అగ్రహార కళాకారుల ఆధ్వర్యంలో, ఈ పరిశ్రమ అధిక స్థాయికి ఎదిగింది. "మాలాప్రభువు", "పాతాల భైరవి" వంటి సినిమాలు ప్రేక్షకులను ఎంతో ఆకట్టుకున్నాయి. 

1980లలో చిరంజీవి వంటి నటులు ఈ పరిశ్రమకు కొత్త జీవం పోసారు. "కంబళీ పట్టు", "అల్లరి నరేష్" వంటి సినిమాలు కొత్త దిశలో మలచాయి. ఇవి యువతను ఆకర్షించడంతో పాటు, సంప్రదాయాలపై అంకితభావాన్ని పంచాయి. 

నేటి రోజుల్లో, టెక్నాలజీ అభివృద్ధి తో, తెలుగు సినిమాలు అంతర్జాతీయ స్థాయిలో గుర్తింపు పొందుతున్నాయి. ప్రతి సంవత్సరం, కొత్త కథలు, కొత్త దిశలను అన్వేషిస్తూ, భారతీయ ప్రేక్షకులను మాత్రమే కాకుండా, అంతర్జాతీయ ప్రేక్షకులను కూడా ఆకర్షిస్తున్నాయి. 

సినిమా ఒక కళ. ఇది సామాజిక సమస్యలను, అభిమతాలను, మరియు తత్వాలను వ్యక్తపరుస్తుంది. అందువల్ల, ఇది ఒక సాధనంగా కాకుండా, ఒక సామాజిక వాస్తవంగా మారింది. 

ప్రస్తుత కాలంలో, తెలుగు సినిమాలు వినోదానికి మించి, భావోద్వేగాలకు, సాంఘిక సందేశాలకు కూడా ప్రధానంగా పనిచేస్తున్నాయి. "కథల పీఠం", "రాజుగారి గది", "శ్రీమంతుడంటే ఇష్టం" వంటి సినిమాలు ఈ దిశలో నూతన అధ్యాయాలను పాఠిస్తున్నాయి.
"""  
print("Summary:", generate_summary(sample_text))

