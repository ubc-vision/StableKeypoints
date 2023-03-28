import torch

class Context_Estimator(torch.nn.Module):
    def __init__(self, tokenizer, text_encoder, num_words = 2, word_size = 768, device="cuda", max_legnth = 77):
        super(Context_Estimator, self).__init__()
        
        self.num_words = num_words
        self.word_size = word_size
        self.device = device
        # the network will be a dilated convolutional layer and a linear layer
        # input will be 4x64x64 image
        self.conv1 = torch.nn.Conv2d(5, 8, kernel_size=3, stride=1, padding=1).to(device)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1).to(device)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1).to(device)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1).to(device)
        
        self.linear = torch.nn.Linear(4*4*64, num_words*word_size).to(device)
        
        text_input = tokenizer(
            "prompt",
            padding="max_length",
            max_length=max_legnth,
            truncation=True,
            return_tensors="pt",
        )
        
        self.baseline_tokens = text_encoder(text_input.input_ids.to(device))[0]

        
    def gaussian_circle(self, pos, size=64, sigma=16):
        """Create a 2D Gaussian circle with a given size, standard deviation, and center coordinates."""
        # import ipdb; ipdb.set_trace()
        _pos = pos*size
        grid = torch.meshgrid(torch.arange(size).to(self.device), torch.arange(size).to(self.device))
        grid = torch.stack(grid, dim=-1)
        dist_sq = (grid[..., 0] - _pos[0])**2 + (grid[..., 1] - _pos[1])**2
        dist_sq = -1*dist_sq / (2. * sigma**2.)
        gaussian = torch.exp(dist_sq)
        return gaussian
        

    def forward(self, image, input_point=None):
        # add a gaussian to the image at the input point
        # gaussian = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3])
        # import ipdb; ipdb.set_trace()
        
        # print("input_point")
        # print(input_point)
        
        # from time import sleep

        circle = self.gaussian_circle(input_point).reshape(1, 1, 64, 64)
        # import matplotlib.pyplot as plt
        # plt.imshow(circle[0,0].cpu().detach().numpy())
        # plt.show()
        # plt.savefig('circle.png')
        # exit()
        # print("sleeping")
        # sleep(10)
        # print("done sleeping")
        image = torch.cat((image, circle), dim=1)
        
        
        x = torch.nn.functional.relu(self.conv1(image))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = x.reshape(-1, self.num_words, self.word_size)
        
        
        output = self.baseline_tokens.clone()
        
        output[:, 1:self.num_words+1, :] = x
        
        
        
        return output
        