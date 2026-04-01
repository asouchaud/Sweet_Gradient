import numpy as np
import sofar as sf
import copy


import numpy as np

def sofa_find(az, sofaile, elv=0, r=0):
    pos = sofaile.SourcePosition[:, :2]  # (N, 2)

    source = np.array([az,elv])

    diff = pos - source

    distance = np.linalg.norm(diff, axis = -1)

    idx = np.argmin(distance)

    return(idx)

def symmetrise_sofa(input_path, output_path, tol=1e-5):
    sofa = sf.read_sofa(input_path)
    new_sofa = copy.deepcopy(sofa)

    positions = sofa.SourcePosition  # (M, 3)
    ir = sofa.Data_IR                # (M, 2, N)

    az = positions[:, 0]
    el = positions[:, 1]
    r  = positions[:, 2]

    new_ir = ir.copy()

    # Normalize azimuths to [0, 360)
    az_norm = np.mod(az, 360)

    for i in range(len(az)):
        azi = az_norm[i]

        # Case 1: angles > 180 → mirror to (360 - azi)
        if azi > 180:
            mirror_az = 360 - azi   
            idx = sofa_find(azi, sofa, elv=el[i])         
            mirror_idx = sofa_find(mirror_az, sofa, elv=el[i])

            # Swap ears
            new_ir[idx, 0, :] = ir[mirror_idx, 1, :]
            new_ir[idx, 1, :] = ir[mirror_idx, 0, :]


        # Case 2: azimuth = 0°
        elif np.isclose(azi, 0.0, atol=tol):
            new_ir[i, 0, :] = ir[i, 0, :]
            new_ir[i, 1, :] = ir[i, 0, :]

        elif np.isclose(azi, 180.0, atol=tol):
            new_ir[i, 0, :] = ir[i, 0, :]
            new_ir[i, 1, :] = ir[i, 0, :]

        # Case 3: 0 < azi ≤ 180 → keep
        else:
            pass

    new_sofa.Data_IR = new_ir
    sf.write_sofa(output_path, new_sofa)

    print(f"Saved symmetrised SOFA to: {output_path}")


if __name__ == "__main__":
    symmetrise_sofa(
        "data\\impulses\\SOFA\\D1_48K_24bit_256tap_FIR_SOFA.sofa",
        "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
    )