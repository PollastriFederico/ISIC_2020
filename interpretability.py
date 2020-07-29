from my_code.nefrologia.grad_cam import GradCam


def explain_eval(self):
    sigm = nn.Sigmoid()
    sofmx = nn.Softmax(dim=1)
    trues = 0
    tr_trues = 0
    acc = 0
    self.n.eval()
    grad_cam = GradCam(self.n, target_layer_names=["7"], use_cuda=True)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None

    start_time = time.time()

    for i, (x, target, img_name) in enumerate(self.eval_data_loader):
        # measure data loading time
        # print("data time: " + str(time.time() - start_time))

        # compute output
        x = x.to('cuda')
        output = torch.squeeze(self.n(x))
        if self.num_classes == 1:
            target = target.to('cuda', torch.float)
            check_output = sigm(output)
            res = (check_output > self.thresh).float()
        else:
            target = target.to('cuda', torch.long)
            check_output = sofmx(output)
            check_output, res = torch.max(check_output, 1)

        tr_target = target * 2
        tr_target = tr_target - 1
        tr_trues += sum(res == tr_target).item()
        trues += sum(res).item()
        acc += sum(res == target).item()

        # gb_model = GuidedBackprop(self.n)
        for j in range(len(x)):
            in_im = Variable(x[j].unsqueeze(0), requires_grad=True)
            # mask = grad_cam(in_im, 0)
            # show_cam_on_image(nefro.denormalize(x[j]), mask, os.path.basename(img_name[j])[:-4] + self.lbl_name + '_cls0')
            mask = grad_cam(in_im, target_index)
            denorm_img = np.asarray(isic.denormalize(x[j].clone()))
            denorm_img = cv2.cvtColor(np.moveaxis(denorm_img, 0, -1), cv2.COLOR_RGB2BGR)
            show_cam_on_image(denorm_img, mask,
                              os.path.basename(img_name[j])[:-4] + '_' + str(target[j].item()))
            cv2.imwrite('/nas/softechict-nas-1/fpollastri/cvpr_GradCam/' + os.path.basename(img_name[j])[:-4] + '.png',
                        np.uint8(255 * denorm_img))

            # gb = gb_model.generate_gradients(in_im, target_index)
            # save_gradient_images(gb, '/nas/softechict-nas-1/fpollastri/nefro_GradCam/' + os.path.basename(img_name[j])[
            #                                                               :-4] + '_gb.png')
            # cam_gb = np.zeros(gb.shape)
            # if not np.isnan(mask).any():
            #     for c in range(0, gb.shape[0]):
            #         cam_gb[c, :, :] = mask
            #     cam_gb = np.multiply(cam_gb, gb)
            # save_gradient_images(cam_gb, '/nas/softechict-nas-1/fpollastri/nefro_GradCam/' + os.path.basename(img_name[j])[
            #                                                                   :-4] + '_cam_gb.png')

        # # measure elapsed time
        # printProgressBar(i + 1, total + 1,
        #                  length=20,
        #                  prefix=f'Epoch {epoch} ',
        #                  suffix=f', loss: {loss.item():.3f}'
        #                  )
    pr = tr_trues / (trues + 10e-5)
    rec = tr_trues / 375
    fscore = (2 * pr * rec) / (pr + rec + 10e-5)
    stats_string = 'Test set = Acc: ' + str(acc / 1000.0) + ' | F1 Score: ' + str(fscore) + ' | Precision: ' + str(
        pr) + ' | Recall: ' + str(rec) + ' | Trues: ' + str(trues) + ' | Correct Trues: ' + str(
        tr_trues) + ' | time: ' + str(time.time() - start_time)
    print(stats_string)


def explain_validation(self):
    sigm = nn.Sigmoid()
    sofmx = nn.Softmax(dim=1)
    trues = 0
    tr_trues = 0
    acc = 0
    self.n.eval()
    grad_cam = GradCam(self.n, target_layer_names=["7"], use_cuda=True)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None

    start_time = time.time()

    for i, (x, target) in enumerate(self.valid_data_loader):
        # measure data loading time
        # print("data time: " + str(time.time() - start_time))

        # compute output
        x = x.to('cuda')
        output = torch.squeeze(self.n(x))
        if self.num_classes == 1:
            check_output = sigm(output)
            res = (check_output > self.thresh).float()
        else:
            check_output = sofmx(output)
            check_output, res = torch.max(check_output, 1)

        # gb_model = GuidedBackprop(self.n)
        for j in range(len(x)):
            in_im = Variable(x[j].unsqueeze(0), requires_grad=True)
            # mask = grad_cam(in_im, 0)
            # show_cam_on_image(nefro.denormalize(x[j]), mask, os.path.basename(img_name[j])[:-4] + self.lbl_name + '_cls0')
            mask = grad_cam(in_im, target_index)
            denorm_img = np.asarray(isic.denormalize(x[j].clone()))
            denorm_img = cv2.cvtColor(np.moveaxis(denorm_img, 0, -1), cv2.COLOR_RGB2BGR)
            show_cam_on_image(denorm_img, mask,
                              str(i * self.batch_size + j) + '_class' + str(res[j].item()))
            cv2.imwrite('/nas/softechict-nas-1/fpollastri/cvpr_GradCam/' + str(i * self.batch_size + j) + '_class' + str(
                res[j].item()) + '.png',
                        np.uint8(255 * denorm_img))

            # gb = gb_model.generate_gradients(in_im, target_index)
            # save_gradient_images(gb, '/nas/softechict-nas-1/fpollastri/nefro_GradCam/' + os.path.basename(img_name[j])[
            #                                                               :-4] + '_gb.png')
            # cam_gb = np.zeros(gb.shape)
            # if not np.isnan(mask).any():
            #     for c in range(0, gb.shape[0]):
            #         cam_gb[c, :, :] = mask
            #     cam_gb = np.multiply(cam_gb, gb)
            # save_gradient_images(cam_gb, '/nas/softechict-nas-1/fpollastri/nefro_GradCam/' + os.path.basename(img_name[j])[
            #                                                                   :-4] + '_cam_gb.png')

        # # measure elapsed time
        # printProgressBar(i + 1, total + 1,
        #                  length=20,
        #                  prefix=f'Epoch {epoch} ',
        #                  suffix=f', loss: {loss.item():.3f}'
        #                  )


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cv2.imwrite('/nas/softechict-nas-1/fpollastri/cvpr_GradCam/' + name + '_cam.png', np.uint8(255 * cam))