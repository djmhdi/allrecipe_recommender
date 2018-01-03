from flask import Flask, render_template, request, jsonify
import pickle


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('submit.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Return a number of recipes (5?), title, weblink
    """
    data = str(request.form['article_body'])
    pred = str(model.predict([data])[0])
    return render_template('recommend.html', article=data, predicted=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
