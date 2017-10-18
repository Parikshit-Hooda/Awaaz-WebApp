#!flask/bin/python
#!/remote/Python-2.7/bin/python

from flask import Flask, jsonify
import pyAudioAnalysis
from pyAudioAnalysis import audioTrainTest as aT 

app = Flask(__name__)

user_input = raw_input("Some input please: ")
print(user_input)


@app.route('/', methods=['GET'])
def get_tasks():
    aT.featureAndTrainRegression("pyAudioAnalysis/data/speechEmotion/", 1, 1, aT.shortTermWindow, aT.shortTermStep, "svm", "pyAudioAnalysis/data/svmSpeechEmotion", False)
    result = aT.fileRegression("pyAudioAnalysis/data/speechEmotion/46.wav", "pyAudioAnalysis/data/svmSpeechEmotion", "svm")
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)