gen_optimizer = optim.Adam(gen.parameters(),lr=0.0002)
disc_optimizer = optim.Adam(disc.parameters(),lr=0.0002)

class Losses():
    def __init__(self):
        super().__init__()
        self.disc_losss = n.BCEWithLogitsLoss()
        self.gen_losss = n.BCEWithLogitsLoss()
        self.vgg_loss = n.MSELoss()
        self.mse_loss = n.MSELoss()
        self.lamda = 0.005
        self.eeta = 0.02 
        
    def calculateLoss(self,discriminator, generator,LR_image, HR_image):

        disc_optimizer.zero_grad()
        generated_output = generator(LR_image.to(device).float())
        fake_data = generated_output.clone()
        fake_label = discriminator(fake_data)

        
        HR_image_tensor = HR_image.to(device).float()
        real_data = HR_image_tensor.clone()
        real_label = discriminator(real_data)
        
        relativistic_d1_loss = self.disc_losss((real_label - torch.mean(fake_label)), torch.ones_like(real_label, dtype = torch.float))
        relativistic_d2_loss = self.disc_losss((fake_label - torch.mean(real_label)), torch.zeros_like(fake_label, dtype = torch.float))      

        d_loss = (relativistic_d1_loss + relativistic_d2_loss) / 2
        d_loss.backward(retain_graph = True)
        disc_optimizer.step()

        fake_label_ = discriminator(generated_output)
        real_label_ = discriminator(real_data)
        gen_optimizer.zero_grad()

        g_real_loss = self.gen_losss((fake_label_ - torch.mean(real_label_)), torch.ones_like(fake_label_, dtype = torch.float))
        g_fake_loss = self.gen_losss((real_label_ - torch.mean(fake_label_)), torch.zeros_like(fake_label_, dtype = torch.float))
        g_loss = (g_real_loss + g_fake_loss) / 2
        
        v_loss = self.vgg_loss(vgg.features[:6](generated_output),vgg.features[:6](real_data))
        m_loss = self.mse_loss(generated_output,real_data)
        generator_loss = self.lamda * g_loss + v_loss + self.eeta * m_loss
        generator_loss.backward()
        gen_optimizer.step()

        return d_loss,generator_loss