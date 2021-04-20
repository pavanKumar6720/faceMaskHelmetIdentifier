step -1 : run classification.bat to classify the images based on 
			1)without mask 2) with mask 3)without helmet 4)with helmet


		#given data is not directly usefull for training the model
		#classification.py crops the faces from the given data and classifies them to use for training the model
		#it saves the cropped faces in output folder
    		

step -2 : run train_detector to generate maskdetector.model and helmetdetector.model

step-3 : run test_detector on testimages/testImage.jpg


output is saved as output.jpg
