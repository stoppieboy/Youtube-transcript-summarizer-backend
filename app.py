from flask import Flask
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

@app.route("/")
def hello():
    return "hello"

@app.route("/get_transcript/<video_id>")
def transcript(video_id):
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    formatter = TextFormatter()
    transcript = formatter.format_transcript(raw_transcript)
    inputs = tokenizer.encode("summarize: "+ transcript, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True)
    result = tokenizer.decode(output[0])

    with open('transcript.txt','w') as file:
        file.write(transcript)
    with open('summarized_transcript.txt','w') as file:
        file.write(result)

    return f"{result}"
