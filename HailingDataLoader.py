import torch
import numpy as np

def collate_fn_hailing(batch):
    '''
    :param batch: batch of (image, label)
            image is (N,6) -> x,y,z,hit,mean_time,std_time
            Note label we only dump direction
    :return: batched_sparse_image, batched_label
            batched_sparse_image is (N, 7) -> x, y, z, batchid, hit, mean_time, std_time
            batched_label is (N,2) -> theta, phi
    '''
    # initiate
    batched_sparse_image        = torch.zeros(size=(0,7), dtype=torch.float)
    batched_label               = torch.zeros(size=(0,2), dtype=torch.float)
    for i, item in enumerate(batch):
        # load
        sparse_image            = item[0]
        label                   = np.asarray(item[1])
        # position                = label[:3]
        direction               = label[3:]
        theta                   = np.arctan2(
            np.sqrt(direction[0]**2+direction[1]**2),
            direction[-1]
        )
        phi                     = np.arctan2(direction[1], direction[0])
        # label                   = np.concatenate(
        #     (
        #         np.reshape(position, newshape=(-1,3)),
        #         np.asarray([[theta, phi]])
        #     ),
        #     axis=1
        # )
        label                   = np.asarray([[theta, phi]])
        label                   = torch.tensor(label, dtype=torch.float).view((-1,2))
        # safety
        if sparse_image.shape[0]==0 or len(sparse_image.shape)<=1:
            sparse_image        = np.asarray([[0,0,0,0,0,0]])
            # print("Zero image truth = "+str(label))
        # split the coords and features
        # print("sparse image = "+str(sparse_image))
        coords                  = torch.tensor(sparse_image[:, :3], dtype=torch.float)
        features                = torch.tensor(sparse_image[:, 3:],  dtype=torch.float).view((-1,3))
        # batchid tensor
        # for debug
        # print("sparse_image = "+str(sparse_image))
        # print("type of sparse_image = "+str(type(sparse_image)))
        # print(sparse_image.shape[0])
        batchids                = torch.ones(sparse_image.shape[0], dtype=torch.float).view((-1,1))
        batchids                *= float(i)
        # cat
        sparse_image            = torch.cat(
            (
                coords,
                batchids,
                features
            ),
            1
        )
        # debug
        # print("sparse_image size = "+str(sparse_image.size()))
        # print("label = "+str(label))
        # print("label size = "+str(label.size()))
        # cat to batch
        batched_sparse_image    = torch.cat(
            (batched_sparse_image, sparse_image),
            0
        )
        batched_label           = torch.cat(
            (batched_label, label),
            0
        )
    return (batched_sparse_image, batched_label.view(-1,2))