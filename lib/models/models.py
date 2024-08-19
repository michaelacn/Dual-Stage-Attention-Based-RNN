import torch 
import torch.nn as nn 


class InputAttention(nn.Module): 
    """
    A torch module representing the Input Attention mechanism in DS-RNN. 

    Attributes
    -----
    T : int 
        Loockback window size  

    n: int 
        Number of exogenous variables
    
    m : int 
        Hidden layer's dimension

    References 
    -----

    Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell: 
        A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction. IJCAI 2017

    """
    def __init__(self, T, n, m): 
        super().__init__()
        self.T = T 
        self.n = n 
        self.m = m 

        self.softmax = nn.Softmax(dim=1) 
        self.tanh = nn.Tanh()
        self.W_e = nn.Linear(2*m, T, bias=False)
        self.U_e = nn.Linear(T, T, bias=False) 
        self.V_e = nn.Linear(T, 1, bias=False) 
        self.LSTM = nn.LSTMCell(n, m) # encoder layer


    def forward(self, x): 
        """
        Compute the forward pass according to DS-RNN Input Attention mechanism. 
        We transform Xt by computing a weight vector obtained by attention. 
        This weight vector encodes the importance of each exogenous variable at time t. 


        Parameters
        -----
        x: torch.tensor (bs, T, n)
            Batch input sequence with T lags and n exogenous features.

        Output 
        -----
        x_tilde: torch.tensor (bs, n, T).
            Stores all the projections.
        """

        batch_size = x.shape[0]
        
        # initialize a projection in Rn for each timestep
        x_tilde = torch.zeros((batch_size, self.m, self.T), device=x.device)
        prev_s = torch.zeros((batch_size, self.m), device=x.device)
        prev_h = torch.zeros((batch_size, self.m), device=x.device)
        
        for t in range(self.T):    
            hs = torch.cat([prev_h, prev_s], dim=1) # (bs, 2*m)
            output_hs = self.W_e(hs) # (bs, T)
            output_hs = output_hs[:, None, :].repeat(1, self.n, 1) # (bs, n, T)
            output_x = self.U_e(x.permute(0, 2, 1)) # (bs, T, n) -> (bs, n, T) -> (bs, n, T)
            tanh_sum = self.tanh(output_hs + output_x) # (bs, n, T)
            e_t = torch.squeeze(self.V_e(tanh_sum), dim=-1) # (bs, n, 1)  ---> (bs, n)
            alpha_t = self.softmax(e_t) # (bs, n) weights  
            x_t_tilde = alpha_t * e_t 

            # encoder 
            h_t, s_t = self.LSTM(x_t_tilde, (prev_h, prev_s)) # (bs, p)
            x_tilde[:, :, t] = h_t

            # update previous hidden state and cell state
            prev_h = h_t 
            prev_s = s_t 
        
        return x_tilde



class TemporalAttentionDecoder(nn.Module): 
    """
    A torch module representing the Temporal Attention mechanism in DS-RNN. 

    Attributes
    -----
    T : int 
        Loockback window size  

    p : int 
        Hidden layer's dimension

    m: int 
        Encoder hidden dimension
    
    References 
    -----

    Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell: 
        A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction. IJCAI 2017

    """

    def __init__(self, T, p, m): 
        super().__init__()
        self.T = T 
        self.p = p 
        self.m = m 
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.W_d = nn.Linear(2*p, m, bias=False) 
        self.U_d = nn.Linear(m, m, bias=False)
        self.V_d = nn.Linear(m, 1, bias=False)
        self.LSTM = nn.LSTMCell(1, p)
        self.W_tilde = nn.Linear(m+1, 1)
        self.fc_1 = nn.Linear(p+m, p) 
        self.fc_2 = nn.Linear(p, 1)

    def forward(self, x, y_known): 
        """
        Compute the forward pass according to DS-RNN Temporal Attention mechanism. 
        We transform ht by computing a weight vector obtained by attention across all different hiddent state vector. 
        This weight vector encodes the importance of each timestep, represented by all ht. 


        Parameters
        -----
        x: torch.tensor (bs, m, T)
            Batch input sequence with T timesteps and n exogenous features.

        y_known: torch.tensor (bs, T) 
            Batch of previous targets.

        Output 
        -----
        Y: torch.tensor (bs, 1).
            Stores batch predictions.
        """
        batch_size = x.shape[0]

        prev_d = torch.zeros((batch_size, self.p), device=x.device) # (bs, p)
        prev_s =torch.zeros((batch_size, self.p), device=x.device)

        for t in range(self.T): 

            ds = torch.cat([prev_d, prev_s], dim=1) #(bs, 2p) 
            output_ds = self.W_d(ds)[:, None, :].repeat(1, self.T, 1) #(bs, T, m)
            output_x = self.U_d(x.permute(0, 2, 1)) # (bs, m, T) -> (bs, T, m) -> (bs, T, m)
            tanh_sum = self.tanh(output_ds + output_x) # (bs, T, m) 
            l_t = torch.squeeze(self.V_d(tanh_sum), dim=2) # (bs, T, 1) -> (bs, T)

            # NOTE : not necessary to repeat, prefered for clarity at first. 
            beta = self.softmax(l_t)[:, None, :].repeat(1, self.p, 1) # (bs, T) -> (bs, 1, T) -> (bs, p, T)
            c_t = torch.sum(beta * x, dim=2) # (bs, m) 

            y_tilde = self.W_tilde(torch.cat([y_known[:, t], c_t], dim=1)) # (bs, m+1) -> (bs, 1)
            d_t, s_t = self.LSTM(y_tilde, (prev_d, prev_s)) #(bs, p)

            # update 
            prev_d = d_t    
            prev_s = s_t 

        # once all time steps have been proceed, use the latest Dt, Ct generated and retrieves y_hat_T
        out1 = self.fc_1(torch.cat([prev_d, c_t], dim=1)) # (bs, p + m) -> (bs, p)
        out2 = self.fc_2(out1) # (bs, 1)
        return out2


class DARNN(nn.Module): 
    """
    A Complete implementation of Dual-stage attention RNN . 

    Attributes
    -----
    T : int 
        Loockback window size  

    n : int 
        Number of exogenous features

    m: int 
        Encoder hidden dimension

    p : int 
        Decoder hidden dimension
    
    References 
    -----

    Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell: 
        A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction. IJCAI 2017

    """
    def __init__(self, T, n, m, p): 
        super().__init__()
        self.T = T 
        self.m = m 
        self.p = p 
        self.n = n 
        self.encoder = InputAttention(T, n, m)
        self.decoder = TemporalAttentionDecoder(T, p, m)
        
    def forward(self, x, y_known): 
        """
        DS-RNN forward pass 


        Parameters
        -----
        x : torch.tensor 
            (bs, T, n)  
        
        y_known : torch.tensor 
            (bs, T)


        Output 
        ----- 
        out : torch.tensor 
            (bs, 1)
        """
        x1 = self.encoder(x) 
        out = self.decoder(x1, y_known) 
        return out 