import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from matplotlib import patches



def inference(domain, u_model, E_model, u_0, v_0):
    domain = domain.clone()

    x_s = torch.linspace(domain[0][0], domain[0][1], 100)
    t_s = torch.linspace(domain[1][0], domain[1][1], 100).requires_grad_(True)

    X, T = torch.meshgrid(x_s, t_s, indexing='ij')

    points = torch.stack((X.flatten(), T.flatten()), dim=1)
    points = points.requires_grad_(True)

    u = u_model(points)

    grad_u = torch.autograd.grad(u.sum(), points, create_graph=True)[0]

    u_x = grad_u[:, 0]
    u_t = grad_u[:, 1]

    E = E_model(u_x.unsqueeze(1))

    U = u.reshape(100, 100)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    im = axs[0].contourf(
        X.detach().numpy(),
        T.detach().numpy(),
        U.detach().numpy(),
        levels=50   
    )

    # bc loss
    bc_loss = torch.mean((U[:, 0]**2) + (U[:, -1]**2))

    # ic loss
    u_t = u_t.reshape(100, 100)

    u_t_0 = u_t[:, 0] 
    ic_loss = torch.mean(((U[:, 0] - u_0(x_s))**2) + ((u_t_0 - v_0(x_s))**2))

    # interior loss
    grad_u = torch.autograd.grad(u.sum(), points, create_graph = True)[0]
    u_x = grad_u[:, 0]
    u_t = grad_u[:, 1]
    u_tt = torch.autograd.grad(u_t.sum(), points, create_graph=True)[0][:, 1]
    E = E_model(u_x.unsqueeze(1)).squeeze()
    flux = E * u_x
    flux_x = torch.autograd.grad(
        flux.sum(), points, create_graph=True
    )[0][:, 0]

    interior_residual = u_tt - flux_x
    loss_pde = torch.mean(interior_residual**2)


    fig.colorbar(im, ax=axs[0])
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    axs[0].set_title(f"u(x,t) | interior loss: {loss_pde.item():.4f} | ic loss: {ic_loss.item():.4f} | bc loss: {bc_loss.item():.4f}")
    rect = patches.Rectangle(
        (0, 0),   
        1,
        2,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    axs[0].add_patch(rect)

    epsilon = u_x.detach().numpy()
    E_vals  = E.detach().numpy()
    
    axs[1].scatter(epsilon, E_vals, s=1)

    axs[1].set_xlabel(r"strain ($\varepsilon = \frac{\partial u}{\partial x}$)")
    axs[1].set_ylabel("E(ε) (prediction)")
    axs[1].set_title("Learned Material Law")


    
    idx = np.argsort(epsilon)
    epsilon_sorted = epsilon[idx]
    E_sorted = E_vals[idx]

    d_eps = np.gradient(epsilon_sorted)
    sigma = np.cumsum(E_sorted * d_eps)
    axs[2].plot(epsilon_sorted, sigma)
    axs[2].set_xlabel("strain")
    axs[2].set_ylabel("stress")
    axs[2].set_title("Stress vs strain")

    plt.tight_layout()
    plt.show()





def smooth(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')

def bounds(tracker, window):
    maxs, mins = [], []
    for i in range(len(tracker) // window):
        segment = tracker[i * window : (i + 1) * window]
        maxs.append(max(segment))
        mins.append(min(segment))
    return maxs, mins

def plot_gradients_and_losses(grad_tracker, residue_tracker, lambda_traker, window=25):

    pde_maxs, pde_mins = bounds(grad_tracker["pde"], window)
    ic_maxs,  ic_mins  = bounds(grad_tracker["ic"],  window)
    bc_maxs,  bc_mins  = bounds(grad_tracker["bc"],  window)

    band_x = np.arange(len(pde_maxs)) * window

    pde_loss = [v[0] for v in residue_tracker.values()]
    ic_loss  = [v[1] for v in residue_tracker.values()]
    bc_loss  = [v[2] for v in residue_tracker.values()]

    g_pde = grad_tracker["pde"]
    g_ic  = grad_tracker["ic"]
    g_bc  = grad_tracker["bc"]

    # ratio_ic = [p / (ic + 1e-8) for ic, p in zip(g_ic, g_pde)]
    # ratio_bc = [p / (bc + 1e-8) for bc, p in zip(g_bc, g_pde)]
    lambda_pde = lambda_traker["pde"]
    lambda_ic = lambda_traker["ic"]
    lambda_bc = lambda_traker["bc"]

    # Raw imbalance (what you actually want to diagnose)
    ratio_ic_raw = [p / (ic + 1e-8) for p, ic in zip(g_pde, g_ic)]
    ratio_bc_raw = [p / (bc + 1e-8) for p, bc in zip(g_pde, g_bc)]

    ratio_ic_scaled = [
    (lp * p) / (li * ic + 1e-8)
    for p, ic, lp, li in zip(g_pde, g_ic, lambda_pde, lambda_ic)
    ]
    ratio_bc_scaled = [
        (lp * p) / (lb * bc + 1e-8)
        for p, bc, lp, lb in zip(g_pde, g_bc, lambda_pde, lambda_bc)
    ]


    plt.figure(figsize=(20, 5))

    # Plot 1: grad
    plt.subplot(1, 5, 1)

    s_x = np.arange(len(smooth(g_pde, window)))
    plt.plot(g_pde, alpha=0.15, linewidth=0.5, color="C0")
    plt.plot(g_ic,  alpha=0.15, linewidth=0.5, color="C1")
    plt.plot(g_bc,  alpha=0.15, linewidth=0.5, color="C2")
    plt.plot(s_x, smooth(g_pde, window), color="C0", linewidth=1.2, label="PDE")
    plt.plot(s_x, smooth(g_ic,  window), color="C1", linewidth=1.2, label="IC")
    plt.plot(s_x, smooth(g_bc,  window), color="C2", linewidth=1.2, label="BC")

    plt.fill_between(band_x, pde_mins, pde_maxs, alpha=0.15, color="C0")
    plt.fill_between(band_x, ic_mins,  ic_maxs,  alpha=0.15, color="C1")
    plt.fill_between(band_x, bc_mins,  bc_maxs,  alpha=0.15, color="C2")

    plt.yscale("log")
    plt.title("Gradient Norms")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Plot 2: losses
    plt.subplot(1, 5, 2)
    plt.plot(pde_loss, label="PDE loss")
    plt.plot(ic_loss,  label="IC loss")
    plt.plot(bc_loss,  label="BC loss")
    plt.yscale("log")   # FIX: log scale so smaller losses are visible
    plt.title("Epochs vs Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Plot 3: ratios
    # Plot 3a: raw gradient ratio
    plt.subplot(1, 5, 3)
    s_x = np.arange(len(smooth(ratio_ic_raw, window)))
    plt.plot(ratio_ic_raw, alpha=0.15, linewidth=0.5, color="C1")
    plt.plot(ratio_bc_raw, alpha=0.15, linewidth=0.5, color="C2")
    plt.plot(s_x, smooth(ratio_ic_raw, window), color="C1", linewidth=1.2, label="PDE/IC")
    plt.plot(s_x, smooth(ratio_bc_raw, window), color="C2", linewidth=1.2, label="PDE/BC")
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1.0)
    plt.yscale("log")
    plt.title("Raw gradient ratios")
    plt.xlabel("Iterations")
    plt.ylabel("g_pde / g_other")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Plot 3b: scaled gradient ratio (should converge to 1)
    plt.subplot(1, 5, 4)
    s_x = np.arange(len(smooth(ratio_ic_scaled, window)))
    plt.plot(ratio_ic_scaled, alpha=0.15, linewidth=0.5, color="C1")
    plt.plot(ratio_bc_scaled, alpha=0.15, linewidth=0.5, color="C2")
    plt.plot(s_x, smooth(ratio_ic_scaled, window), color="C1", linewidth=1.2, label=r"$\lambda_{pde}$PDE / $\lambda_{ic}$IC")
    plt.plot(s_x, smooth(ratio_bc_scaled, window), color="C2", linewidth=1.2, label=r"$\lambda_{pde}$PDE / $\lambda_{bc}$BC")
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1.0)
    plt.yscale("log")
    plt.title("Scaled gradient ratios (→ 1.0 = working)")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\lambda g_{pde}$ / $\lambda g_{other}$")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Plot 4: lambdas
    plt.subplot(1, 5, 5)
    plt.plot(lambda_traker['pde'], label = "$\lambda_{pde}$")
    plt.plot(lambda_traker['ic'], label = "$\lambda_{ic}$", linestyle = "--")
    plt.plot(lambda_traker['bc'], label = "$\lambda_{bc}$")

    plt.legend()
    plt.grid()
    plt.xlabel("epochs/iterations")
    plt.ylabel("$\lambda$")
    plt.title("tracking lambda scaling for handling gradient mismatch")


    plt.tight_layout()
    plt.show()


def grad_norm(loss, models):
    if not isinstance(models, list):
        models = [models]

    params = []
    for m in models:
        params += list(m.parameters())

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True
    )

    norm = 0.0
    for g in grads:
        if g is not None:
            norm += torch.sum(g**2)
    return torch.sqrt(norm)


def train(epochs, optimizer, 
          u_predictor_model, E_predictor_model, 
          train_dataloader, test_dataloader, 
          get_interior, get_initial, get_BC,
          get_interior_residual, get_IC_residue, get_BC_residue, 
          domain,
          u_0, v_0,
          lr_annealing_decay = None, initial_lr = 1e-3, lambda_scaling: bool = False):
    
    epoch_loss_track = []
    epoch_loss_track_test = []
    residue_tracker = {}
    grad_tracker = {
        "pde": [],
        "ic": [],
        "bc": []
    }

    lambda_pde = 1
    lambda_ic = 1
    lambda_bc = 1
    lambda_tracker = {"pde": [], "ic": [], "bc": []}

    for epoch in range(epochs):

        residue_tracker[epoch] = []
        epoch_loss = 0
        epoch_loss_test = 0
        
        epoch_loss_PDE = 0
        epoch_loss_IC = 0
        epoch_loss_BC = 0

        g_pde_epoch = 0
        g_ic_epoch = 0
        g_bc_epoch = 0

        g_pdes = []
        g_ics = []
        g_bcs = []

        for train_data in train_dataloader:
            u_predictor_model.train()
            E_predictor_model.train()

            optimizer.zero_grad()
            interior_data   = get_interior(train_data, domain) 
            IC_data         = get_initial(train_data, domain, u_0, v_0) # u, u_t (exact at t = 0)
            BC_data         = get_BC(train_data, domain) # t at u = 0 and t at u = L


            interior_residue  = get_interior_residual(u_predictor_model, E_predictor_model, interior_data)
            IC_residue        = get_IC_residue(u_predictor_model, IC_data)
            BC_residue        = get_BC_residue(u_predictor_model, BC_data)

            loss_pde = torch.mean(interior_residue**2)
            loss_ic  = torch.mean((IC_residue[0]**2) + (IC_residue[1]**2))
            loss_bc  = torch.mean((BC_residue[0]**2) + (BC_residue[1]**2))

            # gradient mismatach
            models = [u_predictor_model, E_predictor_model]
            g_pde = grad_norm(loss_pde, models)
            g_ic  = grad_norm(loss_ic, models)
            g_bc  = grad_norm(loss_bc, models)

            g_pdes.append(g_pde)
            g_ics.append(g_ic)
            g_bcs.append(g_bc)

            g_pde_epoch += g_pde
            g_ic_epoch  += g_ic
            g_bc_epoch  += g_bc

            # total of iniduvidual residue
            epoch_loss_PDE += loss_pde
            epoch_loss_IC += loss_ic
            epoch_loss_BC += loss_bc

            # total
            if lambda_scaling:
                loss = (lambda_pde * loss_pde) + (lambda_ic * loss_ic) + (lambda_bc * loss_bc)
            else:
                lambda_pde, lambda_ic, lambda_bc = 1, 1, 1
                loss =  loss_pde +  loss_ic +  loss_bc

            epoch_loss += loss
            loss.backward()
            optimizer.step()

            grad_tracker["pde"].append(g_pde.item())
            grad_tracker["ic"].append(g_ic.item())
            grad_tracker["bc"].append(g_bc.item())

        # grad_tracker["pde"].append((g_pde_epoch / len(train_dataloader)).item())
        # grad_tracker["ic"].append((g_ic_epoch / len(train_dataloader)).item())
        # grad_tracker["bc"].append((g_bc_epoch / len(train_dataloader)).item())

        # scaling lambda
        g_pde_mean = np.mean([g.item() for g in g_pdes])
        g_ic_mean  = np.mean([g.item() for g in g_ics])
        g_bc_mean  = np.mean([g.item() for g in g_bcs])
        g_max_mean = max(g_pde_mean, g_ic_mean, g_bc_mean)

        beta = 0.9
        # lambda_pde = beta * lambda_pde + (1 - beta) * (g_max_mean / (g_pde_mean + 1e-8))
        lambda_ic  = beta * lambda_ic  + (1 - beta) * (g_max_mean / (g_ic_mean  + 1e-8))
        lambda_bc  = beta * lambda_bc  + (1 - beta) * (g_max_mean / (g_bc_mean  + 1e-8))

        lambda_pde = float(np.clip(lambda_pde, 0.05, 100.0))
        lambda_ic  = float(np.clip(lambda_ic,  0.05, 100.0))
        lambda_bc  = float(np.clip(lambda_bc,  0.05, 100.0))

        if lambda_scaling:
            lambda_tracker["pde"].append(lambda_pde)
            lambda_tracker["ic"].append(lambda_ic)
            lambda_tracker["bc"].append(lambda_bc)
        else:
            lambda_tracker["pde"].append(1)
            lambda_tracker["ic"].append(1)
            lambda_tracker["bc"].append(1)

        residue_tracker[epoch].extend((epoch_loss_PDE.item()/len(train_dataloader), 
                                    epoch_loss_IC.item()/len(train_dataloader), 
                                    epoch_loss_BC.item()/len(train_dataloader)))

        for test_data in test_dataloader:
            
            u_predictor_model.eval()
            E_predictor_model.eval()

            interior_data   = get_interior(test_data, domain) 
            IC_data         = get_initial(test_data, domain, u_0, v_0) # u, u_t (exact at t = 0)
            BC_data         = get_BC(test_data, domain) # t at u = 0 and t at u = L


            interior_residue  = get_interior_residual(u_predictor_model, E_predictor_model, interior_data)
            IC_residue        = get_IC_residue(u_predictor_model, IC_data)
            BC_residue        = get_BC_residue(u_predictor_model, BC_data)


            loss_pde = torch.mean(interior_residue**2)
            loss_ic  = torch.mean((IC_residue[0]**2) + (IC_residue[1]**2))
            loss_bc  = torch.mean((BC_residue[0]**2) + (BC_residue[1]**2))

            # total
            if lambda_scaling:
                loss = (lambda_pde * loss_pde) + (lambda_ic * loss_ic) + (lambda_bc * loss_bc)
            else:
                lambda_pde, lambda_ic, lambda_bc = 1, 1, 1
                loss =  loss_pde +  loss_ic +  loss_bc
            epoch_loss_test += loss


        epoch_loss = epoch_loss / len(train_dataloader)
        epoch_loss_track.append(epoch_loss.item())

        epoch_loss_test = epoch_loss_test / len(test_dataloader)
        epoch_loss_track_test.append(epoch_loss_test.item())


        if (epoch % 10 == 0) or (epoch == epochs-1):
            print(f"EPOCH: {epoch+1} | train loss: {epoch_loss.item():.4f} | test loss: {epoch_loss_test.item():.4f}")
    
        # change lr
        if lr_annealing_decay:
            if epoch < 30:
                new_lr = initial_lr  
            else:
                new_lr = initial_lr / (1 + 0.05 * (epoch - 10))

            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    return epoch_loss_track, epoch_loss_track_test, residue_tracker, grad_tracker, lambda_tracker
