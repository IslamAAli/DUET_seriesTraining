import numpy as np

def export_data(data, base_path, sensor_prefix, seq_name):
    file_path = base_path+'/'+sensor_prefix+'_'+seq_name+'.csv'

    # Move the tensor from GPU to CPU
    torch_array_cpu = data.cpu()

    # Convert the CPU tensor to a NumPy array
    numpy_array = torch_array_cpu.numpy()

    np.savetxt(file_path, numpy_array, delimiter=',')
