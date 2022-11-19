from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictioning', methods=['POST'])
def home():
    img = request.files('Select File')
    return render_template('Home page.html')


@app.route('/getdata', methods=['post'])
def data():
    name1 = request.form['username']
    print(name1)
    output = str(name1)
    return render_template('Home page.html', text=output)


if __name__ == '__main__':
    app.run()
