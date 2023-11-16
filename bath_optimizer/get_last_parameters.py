import sys
import numpy as np

gij_data = np.loadtxt("last_coupling_params.log")
VSL_data = np.loadtxt("last_interact_params.log")
sys_data = np.loadtxt("last_potential_params.log")

N = gij_data.shape[0]
nsample = 10
if len(sys.argv) == 2:
    nsample = int(sys.argv[1])
avg_gij = np.mean(gij_data[N-nsample:N, 1:], axis=0)
avg_VSL = np.mean(VSL_data[N-nsample:N, 1:], axis=0)
avg_sys = np.mean(sys_data[N-nsample:N, 1:], axis=0)

print("--------------")
print("Used step: from {} to {}".format(sys_data[N-nsample, 0], sys_data[N-1, 0]))


print("--------------")
print("   coupling")
print("--------------")

print("gij bend. sym. = {}".format(avg_gij[0]))
print("gij bend. asym. = {}".format(avg_gij[3]))
print("gij sym. asym. = {}".format(avg_gij[6]))

print("giij bend. sym. = {}".format(avg_gij[1]))
print("giij bend. asym. = {}".format(avg_gij[4]))
print("giij sym. asym. = {}".format(avg_gij[7]))

print("gijj bend. sym. = {}".format(avg_gij[2]))
print("gijj bend. asym. = {}".format(avg_gij[5]))
print("gijj sym. asym. = {}".format(avg_gij[8]))

print("--------------")
print("   interact")
print("--------------")

print("VSL/VLL bend. = {}".format(avg_VSL[0]))
print("VSL/VLL sym. = {}".format(avg_VSL[1]))
print("VSL/VLL asym. = {}".format(avg_VSL[2]))

print("--------------")
print("   system")
print("--------------")
print("kr/kr0 = {}".format(avg_sys[1]))
print("kt/kt0 = {}".format(avg_sys[3]))
