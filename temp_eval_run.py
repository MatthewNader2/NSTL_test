import sklearn

input_data = None
fft_data = input_data.t()
unique_elements = sklearn.externals.array_api_compat.numpy.unique_all(fft_data, 5, 7)