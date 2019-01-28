# GHM_Loss-caffe

1、梯度均衡损失函数，caffe上的实现。

   改写caffe.proto文件：
   
   （1）、增加  message LayerParameter { optional GhmcLossParameter ghmc_loss_param = 160;}
   
   （2）、增加  message GhmcLossParameter {
   
          optional uint32 m = 1 [default = 30];
          
          optional float alpha = 2 [default = 0.0];
        }
        
   （3）、使用：    layer {
   
                    name: "ghmcloss"
                    
                    type: "GhmcLoss"
                    
                    bottom: "fc6"
                    
                    bottom: "label"
                    
                    top: "ghmcloss"
                    
                    loss_weight: 1
                    
                    ghmc_loss_param {
                    
                    m: 30
                    
                    alpha: 0.2
                    
                     }
                     
                  }


2、参考论文：https://arxiv.org/abs/1811.05181


3、pytorch参考代码：https://github.com/libuyu/GHM_Detection
