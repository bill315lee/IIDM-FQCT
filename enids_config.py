

def get_proto_param(datatype, train_date):
    proto_size_t = 10
    if datatype == 'NADC':
        ipdt = 0.5
        batch_size = 50000
        if train_date == '1203':
            stream_size = 2300000
        elif train_date == '1210':
            stream_size = 2100000
        elif train_date == '1216':
            stream_size = 1150000
        else:
            raise NotImplementedError

    elif datatype == 'IDS2017':
        ipdt = 1.0
        batch_size = 20000
        if train_date == 'Tuesday':
            stream_size = 460000
        elif train_date == 'Wednesday':
            stream_size = 660000
        elif train_date == 'Thursday':
            stream_size = 500000
        elif train_date == 'Friday':
            stream_size = 680000
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return stream_size, batch_size, ipdt, proto_size_t


class NADC_Config:

    def __init__(self, datatype, train_date):
        self.a_ipdt = 0.5
        self.epochs = 500
        self.lr = 0.1
        self.alpha = 0.01
        self.t1 = 0.5
        self.t2 = 0.05
        self.core_t = 0.05
        self.store_w = 'True'
        self.loss_guided = 'True'
        self.correctify = 'True'

        stream_size, batch_size, ipdt, proto_size_t = get_proto_param(datatype, train_date)

        self.stream_size = stream_size
        self.batch_size = batch_size
        self.ipdt = ipdt
        self.proto_size_t = proto_size_t



class IDS_Config:

    def __init__(self, datatype, train_date):
        self.a_ipdt = 1.0
        self.epochs = 500
        self.lr = 0.1
        self.alpha = 1
        self.t1 = 0.5
        self.t2 = 0.1
        self.core_t = 0.05
        self.store_w = 'True'
        self.loss_guided = 'True'
        self.correctify = 'True'

        stream_size, batch_size, ipdt, proto_size_t = get_proto_param(datatype, train_date)

        self.stream_size = stream_size
        self.batch_size = batch_size
        self.ipdt = ipdt
        self.proto_size_t = proto_size_t




