all:
	nvcc vector_add.cu -o vector_add
	git add .
	git commit -m "updating"
	git push
