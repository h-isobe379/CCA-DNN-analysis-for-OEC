from chainer import cuda

class DeviceContextManager:
    def __init__(self, device):
        self.device = device
        self.device_obj: Optional[Any] = None

    def __enter__(self):
        try:
            if self.device >= 0:
                self.device_obj = cuda.get_device_from_id(self.device)
                self.device_obj.use()
            else:
                self.device_obj = None
        except Exception as e:
            print(f'Unable to set up the device: {e}')
            self.device_obj = None
        return self.device_obj

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f'An exception occurred: {exc_type.__name__}, {exc_value}')

def move_to_device(*arrays, device):
    if device >= 0:
        return [cuda.to_gpu(array, device) for array in arrays]
    return list(arrays)

def prepare_for_gpu(net, optimizer, device):
    if device >= 0:
        net.to_gpu(device)
    optimizer.use_cleargrads()

