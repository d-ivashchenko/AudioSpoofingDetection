import torch
import torch.nn as nn

from classifier import LCNN
from datasets import AudioClassifierDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from encoder import ConvSSADEncoder
from losses import OCSoftmax, AMSoftmax
from utils import AudioEncoderWorkerModel, AudioEncoderClassifierModel, compute_eer
from worker import Worker

from tqdm import tqdm
import argparse


def train(
    train_path,
    valid_path,
    n_frames=1355,
    output_dim=128,
    base_num_channels=8,
    num_res_blocks=3,
    res_block_kernel_size=11,
    tcn_out_channels=[128, 128],
    encoder_model_name='SSAD_encoder_checkpoint_qmil_k11_bnc8_rb3_tcnoc128-128',
    add_loss="softmax",
    enc_dim=64,
    r_real=0.3,
    r_fake=-0.3,
    alpha=20,
    num_epochs=10,
    batch_size=25
):

    min_val_eer = 1.01

    train_dataset_cls = AudioClassifierDataset(train_path, 'ASVspoof2019.LA.cm.train.trn.txt', format='npz')
    train_data_loader_cls = DataLoader(train_dataset_cls, batch_size=batch_size, shuffle=True)

    val_dataset_cls = AudioClassifierDataset(valid_path, 'ASVspoof2019.LA.cm.dev.trl.txt', format='npz')
    val_data_loader_cls = DataLoader(val_dataset_cls, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    ssad_encoder = ConvSSADEncoder(
        input_dim=1, output_dim=output_dim,
        base_num_channels=base_num_channels,
        num_res_blocks=num_res_blocks, res_block_kernel_size=res_block_kernel_size,
        res_block_padding=1, res_block_stride=2, factor=2,
        tcn_out_channels=tcn_out_channels
    )

    cqcc_worker = Worker(output_dim, 20, n_frames, 112)
    lfcc_worker = Worker(output_dim, 20, n_frames, 500)
    mfcc_worker = Worker(output_dim, 20, n_frames, 500)
    imfcc_worker = Worker(output_dim, 20, n_frames, 500)

    encoder_decoder_model = AudioEncoderWorkerModel(
        encoder=ssad_encoder, workers=[cqcc_worker, lfcc_worker, mfcc_worker, imfcc_worker]
    )

    lcnn_classifier = LCNN(n_frames=n_frames, n_coefs=output_dim, emb_size=enc_dim)

    checkpoint = torch.load(f"{encoder_model_name}.pth", map_location=device)
    encoder_decoder_model.load_state_dict(checkpoint['model_state_dict'])

    encoder_model = ConvSSADEncoder(
        input_dim=1, output_dim=output_dim,
        base_num_channels=base_num_channels,
        num_res_blocks=num_res_blocks, res_block_kernel_size=res_block_kernel_size,
        res_block_padding=1, res_block_stride=2, factor=2,
        tcn_out_channels=tcn_out_channels
    )
    encoder_model.load_state_dict(encoder_decoder_model.encoder.state_dict())

    lcnn_model = AudioEncoderClassifierModel(
        encoder=encoder_model,
        classifier=lcnn_classifier
    )

    criterion = nn.CrossEntropyLoss()

    lcnn_optimizer = torch.optim.Adam(lcnn_model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(lcnn_optimizer, gamma=0.5, last_epoch=- 1, verbose=True)

    if add_loss == "amsoftmax":
        amsoftmax_loss = AMSoftmax(2, enc_dim, s=alpha, m=r_real).to(device)
        amsoftmax_loss.train()
        amsoftmax_optimzer = torch.optim.SGD(amsoftmax_loss.parameters(), lr=0.01)

    if add_loss == "ocsoftmax":
        ocsoftmax = OCSoftmax(enc_dim, r_real=r_real, r_fake=r_fake, alpha=alpha).to(device)
        ocsoftmax.train()
        ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=0.01)

    lcnn_model.to(device)

    writer = SummaryWriter(comment=f'{encoder_model_name}_{add_loss}_emb{enc_dim}')

    for epoch_num in range(num_epochs):
        lcnn_model.train()

        running_loss = 0.

        print('\nEpoch: %d ' % (epoch_num + 1))
        for i, (audio_chunk, labels) in tqdm(enumerate(train_data_loader_cls)):
            audio_chunk = audio_chunk.to(device)
            labels = labels.to(device)

            lcnn_outputs, embs = lcnn_model(audio_chunk)
            lcnn_loss = 0.

            if add_loss == "softmax":
                lcnn_loss = criterion(lcnn_outputs, labels)
                lcnn_optimizer.zero_grad()

                lcnn_loss.backward()
                lcnn_optimizer.step()

            if add_loss == "ocsoftmax":
                lcnn_loss, _ = ocsoftmax(embs, labels)
                lcnn_optimizer.zero_grad()
                ocsoftmax_optimzer.zero_grad()

                lcnn_loss.backward()
                lcnn_optimizer.step()
                ocsoftmax_optimzer.step()

            if add_loss == "amsoftmax":
                outputs, moutputs = amsoftmax_loss(embs, labels)
                lcnn_loss = criterion(moutputs, labels)
                lcnn_optimizer.zero_grad()
                amsoftmax_optimzer.zero_grad()

                lcnn_loss.backward()
                lcnn_optimizer.step()
                amsoftmax_optimzer.step()

            running_loss += lcnn_loss.item()
            if i % 50 == 49:
                last_loss = running_loss / 50  # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_num * len(train_data_loader_cls) + i + 1

                writer.add_scalar('Loss/train', last_loss, tb_x)

                running_loss = 0.

        if (epoch_num + 1) % 10 == 0:
            scheduler.step()

        # Val the model
        lcnn_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (audio_chunk, labels) in enumerate(val_data_loader_cls):
                audio_chunk = audio_chunk.to(device)
                labels = labels.to(device)

                lcnn_outputs, embs = lcnn_model(audio_chunk)

                tb_x = epoch_num * len(train_data_loader_cls) + i + 1

                if add_loss == "softmax":
                    lcnn_loss = criterion(lcnn_outputs, labels)
                    score = nn.functional.softmax(lcnn_outputs, dim=1)[:, 0]
                    writer.add_scalar('Loss/val', lcnn_loss.item(), tb_x)

                elif add_loss == "amsoftmax":
                    outputs, moutputs = amsoftmax_loss(embs, labels)
                    lcnn_loss = criterion(moutputs, labels)
                    score = nn.functional.softmax(outputs, dim=1)[:, 0]
                    writer.add_scalar('Loss/val', lcnn_loss.item(), tb_x)

                elif add_loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(embs, labels)
                    writer.add_scalar('Loss/val', ocsoftmaxloss.item(), tb_x)

                idx_loader.append(labels)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]

            print("\nVal EER: {}".format(val_eer))

            writer.add_scalar('Val EER', val_eer, tb_x)

        if val_eer < min_val_eer:
            min_val_eer = val_eer

            if add_loss == 'softmax':
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': lcnn_model.state_dict(),
                    'lcnn_optimizer_state_dict': lcnn_optimizer.state_dict(),
                    'loss': criterion,
                    'val_eer': val_eer,
                }, f'{encoder_model_name}_{add_loss}_emb{enc_dim}.pth')
            elif add_loss == 'amsoftmax':
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': lcnn_model.state_dict(),
                    'lcnn_optimizer_state_dict': lcnn_optimizer.state_dict(),
                    'loss_optimizer_state_dict': amsoftmax_optimzer.state_dict(),
                    'loss_state_dict': amsoftmax_loss.state_dict(),
                    'val_eer': val_eer
                }, f'{encoder_model_name}_{add_loss}_m{r_real}_emb{enc_dim}.pth')
            elif add_loss == 'ocsoftmax':
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': lcnn_model.state_dict(),
                    'lcnn_optimizer_state_dict': lcnn_optimizer.state_dict(),
                    'loss_optimizer_state_dict': ocsoftmax_optimzer.state_dict(),
                    'loss_state_dict': ocsoftmax.state_dict(),
                    'val_eer': val_eer
                }, f'{encoder_model_name}_{add_loss}_r1_{r_real}_r2_{r_fake}_emb{enc_dim}.pth')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)

    parser.add_argument('--n_frames', type=int, default=1355)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--base_num_channels', type=int, default=32)
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--res_block_kernel_size', type=int, default=11)
    parser.add_argument('--tcn_out_channels', type=int, nargs='+', default=[256])
    parser.add_argument('--encoder_model_name', type=str,
                        default='base_SSAD_encoder_checkpoint')
    parser.add_argument('--add_loss', type=str, default='softmax')
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--r_real', type=float, default=0.8)
    parser.add_argument('--r_fake', type=float, default=0.2)
    parser.add_argument('--alpha', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=25)

    args = parser.parse_args()

    train_path = args.train_path
    valid_path = args.valid_path
    n_frames = args.n_frames
    output_dim = args.output_dim
    base_num_channels = args.base_num_channels
    num_res_blocks = args.num_res_blocks
    res_block_kernel_size = args.res_block_kernel_size
    tcn_out_channels = args.tcn_out_channels
    encoder_model_name = args.encoder_model_name
    add_loss = args.add_loss
    enc_dim = args.enc_dim
    r_real = args.r_real
    r_fake = args.r_fake
    alpha = args.alpha
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    print(f"Path to train: {train_path}")
    print(f"Path to validation: {valid_path}")

    print("n_frames:", n_frames)
    print("output_dim:", output_dim)
    print("base_num_channels:", base_num_channels)
    print("num_res_blocks:", num_res_blocks)
    print("res_block_kernel_size:", res_block_kernel_size)
    print("tcn_out_channels:", tcn_out_channels)
    print("encoder_model_name:", encoder_model_name)
    print("add_loss:", add_loss)
    print("enc_dim:", enc_dim)
    print("r_real:", r_real)
    print("r_fake:", r_fake)
    print("alpha:", alpha)
    print("num_epochs:", num_epochs)

    train(
        train_path,
        valid_path,
        n_frames=n_frames,
        output_dim=output_dim,
        base_num_channels=base_num_channels,
        num_res_blocks=num_res_blocks,
        res_block_kernel_size=res_block_kernel_size,
        tcn_out_channels=tcn_out_channels,
        encoder_model_name=encoder_model_name,
        add_loss=add_loss,
        enc_dim=enc_dim,
        r_real=r_real,
        r_fake=r_fake,
        alpha=alpha,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

