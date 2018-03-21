init:
	pip install -r requirements.txt

gif:
	convert -delay 50 figures/*.png animation.gif

.PHONY: init gif
