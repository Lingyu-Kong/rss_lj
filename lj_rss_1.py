from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
import time
from ase.optimize.precon import Exp, PreconLBFGS
import numpy as np
import wandb
import argparse
calc=LennardJones(rc=500)
def lj_relax(atom, no_stress=True, no_traj=False, force_tol=0.01, stress_tol=0.05, log_file=None, max_steps=300):    
    atom.calc=calc
    if no_stress:
        optimizer = BFGS(atom, logfile=log_file)
    else:
        optimizer = PreconLBFGS(atom, precon=Exp(3), use_armijo=True, logfile=log_file, master=True)    
    traj = []
    if not no_traj:
        def build_traj():
            traj.append(atom.copy())
        optimizer.attach(build_traj) 
    
    if no_stress:
        optimizer.run(fmax=0.0001, steps=max_steps)
    else:
        optimizer.run(fmax=force_tol, smax=stress_tol, steps=max_steps)

    if no_traj:
        return atom.copy()#(atom.get_positions(),atom.get_potential_energy())
    else:
        return traj

def generate_structure(N, scale, minsep):

    def recurrent(N, cur, pos, max_attempts = 10):
        if cur == N:
            return True
        for _ in range(max_attempts):
            pos[cur] = np.random.uniform(-scale, scale, 3)
            if cur > 0:
                too_close = False
                for pre in range(0, cur):
                    dist = np.linalg.norm(pos[pre] - pos[cur])
                    if dist < minsep:
                        too_close = True 
                        break
                if too_close:
                    continue
            
            success = recurrent(N, cur + 1, pos)
            if success:
                return True
        return False

    pos = np.random.uniform(-scale, scale, (N, 3))
    count = 0
    while True:
        count += 1
        results = recurrent(N, 0, pos)
        if results:
            return pos
        if count > 100:
            break
    return None
                
def random_generate_structure(N, scale):
    pos = np.random.uniform(-scale, scale, (N, 3))
    return pos            
                        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--natoms", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--minsep", type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default="lj_rss")
    parser.add_argument("--total_trajectories", type=int, default=1000)
    parser.add_argument("--random_start", action="store_true", default=False)
    parser.add_argument("--seq_num", type=int, default=0)
    args = parser.parse_args()
    if args.wandb:
        wandb.init(project="rss lj")

    initial_xyz = []
    for i in range(args.total_trajectories):
        if not args.random_start:
            initial_xyz.append(generate_structure(args.natoms, args.scale, args.minsep))
        else:
            initial_xyz.append(random_generate_structure(args.natoms, args.scale))

    min_energy = [0.0]
    num_dft = [0]

    positions, energies = [], []
    for i in range(args.total_trajectories):
        atom=Atoms('Ar'+str(args.natoms),positions=initial_xyz[i])
        traj = lj_relax(atom)
        min_energy.append(min(min_energy[-1], atom.get_potential_energy()))
        num_dft.append(len(traj) + num_dft[-1])

        e = atom.get_potential_energy()
        for _atom in traj:
            positions.append(_atom.positions)
            energies.append(e)
        if wandb:
            wandb.log({"min energy": min_energy[-1]})

        if i==99:
            np.savez("rss_results/rss_tmp_"+str(args.natoms)+".npz", min_energy=min_energy, num_dft=num_dft, positions=positions, energies=energies)

    if args.wandb:
        data = [[x, y] for (x, y) in zip(min_energy, num_dft)]
        table = wandb.Table(data=data, columns = ["minimum energy", "num of dft"])
        wandb.log({"Random structure search" : wandb.plot.scatter(table, "minimum energy", "num of dft")})
    if not args.random_start:
        np.savez("rss_results/rss_"+str(args.natoms)+" "+str(args.seq_num)+".npz", min_energy=min_energy, num_dft=num_dft, positions=positions, energies=energies)
    else:
        np.savez("rss_results/rss_ranbegin_"+str(args.natoms)+" "+str(args.seq_num)+".npz", min_energy=min_energy, num_dft=num_dft, positions=positions, energies=energies)
    print(str(args.natoms)+" done")


