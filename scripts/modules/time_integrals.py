
import pynlin.fiber
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.constellations


def do_time_integrals(a_chan, fiber, wdm, pulse, overwrite):
    partial_collisions_margin = 2
    points_per_collision = 10

    pynlin.nlin.time_integrals_all_b_chans(
        wdm,
        fiber,
        a_chan,
        pulse,
        "results/results.h5",
        overwrite=overwrite,
        points_per_collision=points_per_collision,  # kwargs
        use_multiprocessing=True,
        partial_collisions_margin=partial_collisions_margin,
        speedup_pulse_propagation=True
    )
    return 0
