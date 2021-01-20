//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object

var startButton = document.getElementById('startButton');
var stopButton = document.getElementById('stopButton');
var recordingsList = document.getElementById('recordingsList');

const imageFolder = '/static/img/'
var imageList = ['girl_lighthouse.jpg', 'test_image.jpg', 'tropical_beach.jpg']
var imageIdx = 0

//add events to those 2 buttons
startButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);


/**
 * Initializes the program, set up the variables
 */
function init() {
	var constraints = { audio: true, video: false };

	// initialize the recorder
	navigator.mediaDevices.getUserMedia(constraints).then(stream => {
		console.log('Initialize the recorder');
		var AudioContext = window.AudioContext || window.webkitAudioContext;
		var audioCtx = new AudioContext();

		/* assign the gumStream for later use, when stop recording */
		gumStream = stream;

		input = audioCtx.createMediaStreamSource(stream);

		rec = new Recorder(input, { numChannels: 1 })

		startButton.disabled = false;
	}).catch(err => {
		alert('Initialize the recorder failed.\n'
			+ 'Either your brower is not support voice recording.\n'
			+ 'Or you disable the accesee to your microphone');
	});

	showImageWithCurrentIdx();
}

/* Call back function when user clicks the recording button */
function startRecording() {
	console.log('start record');
	rec.record();
	stopButton.disabled = false;
	startButton.disabled = true;
}

/* Call back function when user clicks the stop button */
function stopRecording() {
	console.log('stop record')

	rec.stop();

	stopButton.disabled = true;
	startButton.disabled = false;

	createDownloadLink();
	rec.clear();
}

/* Creates the link to download the recorded voice */
function createDownloadLink() {
	rec && rec.exportWAV(blob => {
		var url = URL.createObjectURL(blob);
		console.log('audio link: ' + url);
		var li = document.createElement('li');
		var au = document.createElement('audio');
		var hf = document.createElement('a');

		au.controls = true;
		au.src = url;
		hf.href = url;
		hf.download = new Date().toISOString() + '.wav';
		hf.innerHTML = hf.download;
		li.appendChild(au);
		li.appendChild(hf);
		var upload = document.createElement('a');
		upload.href = "#";
		upload.innerHTML = "Recognize it!";
		upload.addEventListener("click", function (event) {
			var xhr = new XMLHttpRequest();
			xhr.onload = function (e) {
				if (this.readyState === 4) {
					console.log("Server returned: ", xhr.response);
					window.location.href = './demo'
				}
			};
			var fd = new FormData();
			blob.type = 'multipart/form-data';
			fd.append('audio', blob, 'audio.wav');
			fd.append('image_name', imageList[imageIdx]);
			xhr.open('POST', '/audio-data', true);
			xhr.send(fd);
		});
		li.appendChild(document.createTextNode(" "))//add a space in between
		li.appendChild(upload)//add the upload link to li
		recordingsList.appendChild(li);
	});
}

/** Displays the previous image */
function showPrevImage() {
	// Decrease the index by 1, round up to the last image if the index become -1
	const numImages = imageList.length
	imageIdx = (imageIdx - 1 + numImages) % numImages;
	showImageWithCurrentIdx();
  }
  
/** Displays the next image */
function showNextImage() {
	/* Increase the index by 1, round up to the first image if the index is
	   larger than the number of images */
	const numImages = imageList.length
	imageIdx = (imageIdx + 1) % numImages;
	showImageWithCurrentIdx();
  }
  
/** Displays the image with the current index */
function showImageWithCurrentIdx() {
	const imagePath = imageFolder + imageList[imageIdx];
	const imageElement = document.getElementById('image');
	imageElement.src = imagePath;
}