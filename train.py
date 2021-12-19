import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.model_select import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR

def train(opt, data_loader, model, visualizer):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    step_accu = 0
    for epoch in range(model.s_epoch+1, opt.e_epoch+1):
        epoch_start_time = time.time()
        iter_batch = 0
        for _, data in enumerate(dataset):
            iter_start_time = time.time()
            iter_batch     += opt.batchSize
            step_accu      += opt.batchSize

            model.set_input(data)
            model.train_update()

            if step_accu % opt.display_freq == 0:
                results = model.get_current_visuals()
                psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
                print('PSNR on Train = %f' % psnrMetric)
                visualizer.display_current_results(results, epoch)

            if step_accu % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, iter_batch, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(iter_batch)/dataset_size, opt, errors)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, step_accu))
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.e_epoch, time.time()-epoch_start_time))

        #if epoch > opt.e_epoch:
        #	model.update_learning_rate()

if __name__ == '__main__':
    opt 					= TrainOptions().GetOption()
    data_loader 			= CreateDataLoader(opt)
    model					= create_model(opt)
    visualizer 				= Visualizer(opt)
    train(opt, data_loader, model, visualizer)
    print('End Training')
