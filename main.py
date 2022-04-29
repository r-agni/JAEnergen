
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_15MIN_PER_HOUR = 4
N_15MIN_PER_DAY = 96


def country_demand(
        country_code: str,
) -> np.ndarray:
    """Return demand in specified country.
    Parameters
    ----------
    country_code : str
        Country code (e.g., "AT" for Austria, "DE" for Germany).
    Returns
    -------
    gw_demand : numpy.ndarray
        Array of shape `(n_15min,)` giving the demand (GW) over `n_15min`
        successive 15-minute intervals.
    """
    df = pd.read_csv("time_series_15min_singleindex.csv")
    first_15min_index = 5
    last_15min_index = 34949  # 200933
    print("\nFirst 15-minute interval begins at UTC",
          pd.to_datetime(df["utc_timestamp"]).iloc[first_15min_index])
    print("Last 15-minute interval begins at UTC",
          pd.to_datetime(df["utc_timestamp"]).iloc[last_15min_index - 1])
    gw_demand = 1e-3 * df[
        f"{country_code}_load_actual_entsoe_transparency"
    ][first_15min_index : last_15min_index].to_numpy()
    print(f"Number of 15-minute intervals = {gw_demand.size}"
          f" ({gw_demand.size / 96} days, {gw_demand.size / 672} weeks)")
    print(f"Average demand = {np.mean(gw_demand):.1f} GW")
    print(f"Maximum demand = {np.max(gw_demand):.1f} GW")
    print(f"Minimum demand = {np.min(gw_demand):.1f} GW")
    return gw_demand


def country_supply_clean(
        country_code: str,
) -> np.ndarray:
    """Return solar + wind supply in specified country.
    Parameters
    ----------
    country_code : str
        Country code (e.g., "AT" for Austria, "DE" for Germany).
    Returns
    -------
    gw_supply_clean : numpy.ndarray
        Array of shape `(n_15min,)` giving the solar + wind supply (GW) over
        `n_15min` successive 15-minute intervals.
    """
    df = pd.read_csv("time_series_15min_singleindex.csv")
    first_15min_index = 5
    last_15min_index = 34949  # 200933
    print("\nFirst 15-minute interval begins at UTC",
          pd.to_datetime(df["utc_timestamp"]).iloc[first_15min_index])
    print("Last 15-minute interval begins at UTC",
          pd.to_datetime(df["utc_timestamp"]).iloc[last_15min_index - 1])
    gw_supply_clean = 1e-3 * df[
        f"{country_code}_solar_generation_actual"
    ][first_15min_index : last_15min_index].fillna(0).to_numpy()
    gw_supply_clean += 1e-3 * df[
        f"{country_code}_wind_generation_actual"
    ][first_15min_index : last_15min_index].fillna(0).to_numpy()
    print(f"Number of 15-minute intervals = {gw_supply_clean.size}")
    print(f"Average solar + wind supply = {np.mean(gw_supply_clean):.1f} GW")
    print(f"Maximum solar + wind supply = {np.max(gw_supply_clean):.1f} GW")
    print(f"Minimum solar + wind supply = {np.min(gw_supply_clean):.1f} GW")
    return gw_supply_clean


def average_demand_over_interval(
        gw_demand: np.ndarray,
        n_15min_per_interval: int,
        moving_average: bool = False,
) -> np.ndarray:
    """Return demand averaged over given number of 15-minute intervals.
    Parameters
    ----------
    gw_demand : numpy.ndarray
        Array of shape `(n_15min,)` giving the demand (GW) over `n_15min`
        successive 15-minute intervals.
    n_15min_per_interval : int
        Number of 15-minute intervals over which the demand is to be averaged.
    moving_average : bool
        If `True`, return moving average over specified number of 15-minute
        intervals. Otherwise return average over disjoint blocks of
        specified number of 15-minute intervals.
    Returns
    -------
    gw_demand_after_averaging : numpy.ndarray
        Array of shape `(n_15min,)` giving the demand (GW) over `n_15min`
        successive 15-minute intervals, averaged over blocks of
        `n_15min_per_interval` such intervals.
    """
    if moving_average:
        return np.convolve(
            gw_demand, np.ones(n_15min_per_interval) / n_15min_per_interval
        )[n_15min_per_interval - 1 :]
    n_15min = gw_demand.size
    n_whole_intervals = n_15min // n_15min_per_interval
    idx = n_whole_intervals * n_15min_per_interval
    gw_demand_after_averaging = np.zeros(n_15min)
    gw_demand_after_averaging[: idx] = np.repeat(
        np.reshape(
            gw_demand[: idx], (n_whole_intervals, n_15min_per_interval)
        ).mean(1),
        n_15min_per_interval
    )
    if idx < n_15min:
        gw_demand_after_averaging[idx :] = gw_demand[idx :].mean()
    return gw_demand_after_averaging


def required_storage(
        gw_demand: np.ndarray,
        gw_supply: np.ndarray,
) -> float:
    """Return storage requirement.
    Parameters
    ----------
    gw_demand : numpy.ndarray
        Array of shape `(n_15min,)` giving the demand (GW) over `n_15min`
        successive 15-minute intervals.
    gw_supply : numpy.ndarray
        Array of shape `(n_15min,)` giving the supply (GW) over `n_15min`
        successive 15-minute intervals.
    Returns
    -------
    gwh_needed : float
        Required storage (GWh) to within 10% (computed by starting at 0.1 GWh
        and increasing by 10% whenever insufficient).
    """
    n_15min = gw_demand.size
    gwh_needed = 0.1
    while True:
        gwh_stored = np.zeros(n_15min + 1)
        gwh_stored[0] = gwh_needed
        enough_storage = True
        for t in range(n_15min):
            gwh_deficit = (gw_demand[t] - gw_supply[t]) / N_15MIN_PER_HOUR
            if gwh_stored[t] < gwh_deficit:
                print(f"\r{gwh_needed:.1f} GWh storage not enough", end="")
                gwh_needed *= 1.1
                enough_storage = False
                break
            gwh_stored[t + 1] = min(gwh_stored[t] - gwh_deficit, gwh_needed)
        if enough_storage:
            print(f"\n{gwh_needed:.1f} GWh storage is enough!")
            return gwh_needed


def plot_country_demand(
        country_code: str,
        all_n_hours_to_average,
        first_day_to_plot: int = 0,
        n_days_to_plot: int = 10,
        moving_average: bool = False,
) -> None:
    """Plot demand in country averaged over various time intervals.
    Parameters
    ----------
    country_code : str
        Country code (e.g., "AT" for Austria, "DE" for Germany).
    all_n_hours_to_average : list[int]
        List specifying all the averaging intervals in hours.
    first_day_to_plot : int
        Index of first day for which to plot the demand after averaging.
    n_days_to_plot : int
        Number of days for which to plot the demand after averaging.
    moving_average : bool
        If `True`, plot moving average. Otherwise plot blockwise average.
    """
    first_idx = first_day_to_plot * N_15MIN_PER_DAY
    last_idx = first_idx + n_days_to_plot * N_15MIN_PER_DAY
    fig, axs = plt.subplots()
    gw_demand = country_demand(country_code)
    days_since_start = np.arange(gw_demand.size) / N_15MIN_PER_DAY
    # axs.plot(
    #     days_since_start[first_idx : last_idx],
    #     gw_demand[first_idx : last_idx],
    #     label="No averaging", color="k", linestyle=":"
    # )
    for n_hours_to_average in all_n_hours_to_average:
        gw_demand_after_averaging = average_demand_over_interval(
            gw_demand,
            n_hours_to_average * N_15MIN_PER_HOUR,
            moving_average
        )
        axs.plot(
            days_since_start[first_idx : last_idx],
            gw_demand_after_averaging[first_idx : last_idx],
            label=f"{n_hours_to_average}-hour averaging", linewidth=2
        )
    gw_supply_clean = country_supply_clean(country_code)
    axs2 = axs.twinx()
    axs2.plot(
        days_since_start[first_idx : last_idx],
        gw_supply_clean[first_idx : last_idx],
        label="Solar + wind", color="C9", linestyle="--"
    )
    axs.set_xlabel("Time in days")
    axs.set_ylabel("Demand (GW) averaged over a number of hours")
    axs2.set_ylabel("Solar + wind supply (GW)")
    axs.set_title(f"Country code = {country_code}")
    # axs.grid(True)
    axs.legend()
    plt.show()


def plot_country_clean_supply(
        country_code: str,
        all_n_hours_to_average,
        first_day_to_plot: int = 0,
        n_days_to_plot: int = 10,
        moving_average: bool = False,
) -> None:
    """Plot clean supply in country averaged over various time intervals.
    Parameters
    ----------
    country_code : str
        Country code (e.g., "AT" for Austria, "DE" for Germany).
    all_n_hours_to_average : list[int]
        List specifying all the averaging intervals in hours.
    first_day_to_plot : int
        Index of first day for which to plot the supply after averaging.
    n_days_to_plot : int
        Number of days for which to plot the supply after averaging.
    moving_average : bool
        If `True`, plot moving average. Otherwise plot blockwise average.
    """
    first_idx = first_day_to_plot * N_15MIN_PER_DAY
    last_idx = first_idx + n_days_to_plot * N_15MIN_PER_DAY
    fig, axs = plt.subplots()
    gw_supply_clean = country_supply_clean(country_code)
    days_since_start = np.arange(gw_supply_clean.size) / N_15MIN_PER_DAY
    axs.plot(
        days_since_start[first_idx : last_idx],
        gw_supply_clean[first_idx : last_idx],
        label="No averaging", color="k", linestyle=":"
    )
    for n_hours_to_average in all_n_hours_to_average:
        gw_supply_clean_after_averaging = average_demand_over_interval(
            gw_supply_clean,
            n_hours_to_average * N_15MIN_PER_HOUR,
            moving_average
        )
        axs.plot(
            days_since_start[first_idx : last_idx],
            gw_supply_clean_after_averaging[first_idx : last_idx],
            label=f"{n_hours_to_average}-hour averaging", linewidth=2
        )
    axs.set_xlabel("Time in days")
    axs.set_ylabel("Solar + wind supply (GW)")
    axs.set_title(f"Country code = {country_code}")
    # axs.grid(True)
    axs.legend()
    plt.show()


def cmd_line_args(
) -> argparse.Namespace:
    """Return command-line arguments.
    Returns
    -------
    args : argparse.Namespace
        Object containing values of command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--country_code",
        default="DE",
        help="Country code (default: DE)"
    )
    parser.add_argument(
        "-a", "--all_n_hours_to_average",
        nargs="+", type=int, default=[24, 168, 8736],
        help="All numbers of hours to average over (default: 24 168 8736)"
    )
    parser.add_argument(
        "-f", "--first_day_to_plot",
        type=int, default=0,
        help="Index of first day for which to plot the demand (default: 0)"
    )
    parser.add_argument(
        "-d", "--n_days_to_plot",
        type=int, default=0,
        help="Number of days for which to plot the demand (default: 0)"
    )
    parser.add_argument(
        "-m", "--moving_average",
        action="store_true",
        help="If specified, plot the moving average"
    )
    return parser.parse_args()


def run_main(
) -> None:
    """Run main function.
    """
    args = cmd_line_args()
    print(vars(args))
    if args.n_days_to_plot > 0:
        plot_country_demand(
            country_code=args.country_code,
            all_n_hours_to_average=args.all_n_hours_to_average,
            first_day_to_plot=args.first_day_to_plot,
            n_days_to_plot=args.n_days_to_plot,
            moving_average=args.moving_average,
        )
        plot_country_clean_supply(
            country_code=args.country_code,
            all_n_hours_to_average=args.all_n_hours_to_average,
            first_day_to_plot=args.first_day_to_plot,
            n_days_to_plot=args.n_days_to_plot,
            moving_average=args.moving_average,
        )
    gw_demand = country_demand(args.country_code)
    print(f"Required peak supply capacity without averaging"
          f" = {np.max(gw_demand):.1f} GW")
    # Evaluate required peak supply capacity with averaging.
    for n_hours_to_average in args.all_n_hours_to_average:
        gw_supply = average_demand_over_interval(
            gw_demand,
            n_hours_to_average * N_15MIN_PER_HOUR,
            args.moving_average
        )
        print(f"\nWith averaging over {n_hours_to_average} hours:")
        print(f"Required peak supply capacity = {np.max(gw_supply):.1f} GW")
        _ = required_storage(gw_demand, gw_supply)
    # Evaluate required peak supply capacity with clean + dirty.
    gw_supply_clean = country_supply_clean(args.country_code)
    for multiple in (1, 10):
        for gw_supply_dirty in (30.0,):
            print(f"\nWith {multiple}x current clean"
                  f" and {gw_supply_dirty} GW dirty:")
            gw_supply = multiple * gw_supply_clean + gw_supply_dirty
            _ = required_storage(gw_demand, gw_supply)


if __name__ == "__main__":
    run_main()
