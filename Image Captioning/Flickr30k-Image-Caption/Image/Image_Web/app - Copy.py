from flask import Flask, render_template, redirect, request
from gtts import gTTS
import os

import Caption_It_1



# __name__ == __main__
app = Flask(__name__)


@app.route('/')
def hello():
	return render_template("index - Copy.html")


@app.route('/', methods= ['POST'])
def marks():
	if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)
		path2 = "./static/{}".format(f.filename) + ".mp3"

		#caption = Caption_it.caption_this_image(path)
		caption = Caption_It_1.runModel(path)
		output = gTTS(text = caption, lang = 'en',slow = False)
		output.save(path2)

		result_dic = {
		'image' : path,
		'caption' : caption,
		'sound'   : path2
		}

	return render_template("index - Copy.html", your_result =result_dic)

if __name__ == '__main__':
	# app.debug = True
	# due to versions of keras we need to pass another paramter threaded = Flase to this run function
	app.run(debug = True)
