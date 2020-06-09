from flask import Flask
import flask
import tensorflow as tf
# import five_words as five

#load_path = "/home/garima/Desktop/lyric model/keras_model_word_hdf5"


model=tf.keras.models.load_model('./keras_model_word.hdf5')
app = Flask(__name__, template_folder="templates")
run_with_ngrok(app) 


def infer(prime_text):
    prime_text = prime_text.lower().split(" ")[:55]
    lyricmodel = predict_n(model,prime_text,50)
    #result = lyricmodel.test(prime_text)
    return lyricmodel

@app.route('/', methods=['GET'])
def webpage():
    return flask.render_template('index.html')

@app.route('/lyrics', methods = ['POST'])
def search(): 
    content = flask.request.get_json(silent = True)
    input_text = content['search_text']
    print(input_text)
    generated = infer(input_text)
    return flask.jsonify({'generated': generated})

@app.route('/test', methods=['GET'])
def test():
    return flask.jsonify({'ping': 'ping_data'})
app.run()
