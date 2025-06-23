import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import os
from typing import Optional


def Qplot(
        train_state,
        networks,
        plotV: bool = True, 
        levels: int = 100, 
        title: str = "", 
        savefig: Optional[str] = None):
    # /!\ WORKS ONLY WITH 1D OBSERVATION AND 1D ACTION SPACES,
    # AND BOTH EQUAL TO [-1, 1]
    ax = plt.figure().add_subplot(projection='3d')
    K = 100
    S, A = np.meshgrid(np.linspace(-1, 1, K), np.linspace(-1, 1, K))
    S_flat = np.expand_dims(S.flatten(), 1)
    A_flat = np.expand_dims(A.flatten(), 1)
    s_range = np.linspace(-1, 1, K)
    a_range = np.linspace(-1, 1, K)

    Q = networks.critic.apply(train_state.params_critic, S_flat, A_flat)[0]
    Q = np.array(Q).reshape((K, K))
    if plotV:   
        V = networks.value.apply(train_state.params_value, S_flat)[0]
        V = np.array(V).reshape((K, K))
        
    max_points_x = []
    max_points_y = []
    max_points_z = []
    argmaxes = np.argmax(Q, axis=0)
    for i in range(K):
        max_points_x.append(s_range[i])
        max_points_y.append(a_range[argmaxes[i]])
        max_points_z.append(Q[argmaxes[i], i])

    ax.contour(S, A, Q, levels=levels, cmap=cm.coolwarm)
    if plotV:
        ax.contour(S, A, V, levels=levels, cmap=cm.copper)
    ax.plot(max_points_x, max_points_y, max_points_z, linewidth=2)
    ax.set_xlabel("s")
    ax.set_ylabel("a")
    ax.set_title(title)
    ax.view_init(elev=90., azim=-90.)

    if savefig is None:
        fm = plt.get_current_fig_manager()
        fm.window.showMaximized()
        plt.show()
    else:
        # make sure that folder exists
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig)
        plt.close()