import warnings
warnings.filterwarnings("ignore")

import fasttext
import streamlit as st

from PIL import Image


def load_model():
    model = fasttext.train_supervised("data/train_clean.txt", lr = 0.3, epoch = 25, wordNgrams = 2)
    print("[INFO] loaded face mask detector model")
    #model.save_model("model.bin")
    #model = fasttext.load_model("model.bin")
    return model


def predict_humor(model: fasttext.FastText._FastText, sentence: str) -> str:
    predictions = model.predict(sentence)
    label = predictions[0][0].split("__label__")[1]
    if label == "not_humorous":
        label = label.replace("_", " ").title()
    else:
        label = label.title()
    confidence = predictions[1][0]
    return "{} ({:.2f}% confident)".format(label, confidence * 100)


def main():
    image = Image.open("laughing-emoji.jpg")
    st.image(image, use_column_width = True)

    st.write("""
    # Humor Detection Web App

    This app leverages Facebook Research's [fastText](https://fasttext.cc/) library to predict if your sentence is humorous or not!
    """)
    model = load_model()
    user_sentence = st.text_input("Check here if your sentence is humorous or not", "I am a master of tearable puns but only on paper")
    print("user input: ", user_sentence)
    st.write("Your sentence is:", predict_humor(model, user_sentence))


if __name__ == "__main__":
    main()

