from lightpy import (
    run_single_slit_experiment,
    run_double_slit_experiment,
    run_circular_aperture_experiment,
    run_grating_experiment,
    run_all_checks
)

def main():

    run_all_checks("single_slit_basic.json")
    run_single_slit_experiment("single_slit_basic.json")

    run_all_checks("double_slit_interference.json")
    run_double_slit_experiment("double_slit_interference.json")

    run_all_checks("circular_aperture_airy.json")
    run_circular_aperture_experiment("circular_aperture_airy.json")

    run_all_checks("diffraction_grating.json")
    run_grating_experiment("diffraction_grating.json")

if __name__ == '__main__':
    main()






