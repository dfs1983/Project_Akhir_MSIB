# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import RandomRotation, RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import applications
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from PIL import Image
from fungsi import make_model

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG', '.png', '.PNG']
app.config['UPLOAD_PATH']        = './static/img/uploads/'

model = None

NUM_CLASSES = 5
Wayang5_classes = ["bagong", "cepot", "gareng", "petruk", "semar"] 

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename)
	
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = './static/img/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image         = Image.open('.' + gambar_prediksi)
			
			# Mengubah Ukuran Gambar
			test_image_resized = test_image.resize((128, 128))
			
			# Konversi Gambar ke Array
			image_array        = np.array(test_image_resized)
			test_image_x       = (image_array / 255) 
			test_image_x       = np.array([image_array])

			test_image_x = tf.image.resize(test_image_x, IMG_SIZE)

			
			# Prediksi Gambar
			y_pred_test_single         = model.predict(test_image_x)
			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
			hasil_prediksi = Wayang5_classes[y_pred_test_classes_single[0]]
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})

# =[Main]========================================		

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = make_model()
	model.load_weights("model_Wayang5_cnn_tf.h5")

	# Run Flask di localhost 
	run_with_ngrok(app)
	app.run()

	
	


