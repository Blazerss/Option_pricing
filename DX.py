import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_year_deltas(date_list, day_count=365.):
    """ Return vector of floats with day deltas in year fractions.
    Initial value normalized to zero.
    Parameters
    ==========
    date_list: list or array
    collection of datetime objects
    day_count: float
    number of days for a year
    (to account for different conventions)
    Results
    =======
    delta_list: array
    year fractions
    """
    start = date_list[0]
    delta_list = [(date - start).days / day_count
                  for date in date_list]
    return np.array(delta_list)


class constant_short_rate(object):
    """ Class for constant short rate discounting.
    Attributes
    ==========
    name: string
       name of the object
    short_rate: float (positive)
       constant rate for discounting
    Methods
    =======
    get_discount_factors:
        get discount factors given a list/array of datetime objects or year fractions
    """

    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError('Short rate negative.')
            # this is debatable given recent market realities

    def get_discount_factors(self, date_list, dtobjects=True):
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * np.sort(-dlist))
        return np.array((date_list, dflist)).T


class market_environment(object):
    """
    Class to model a market environment relevant for valuation.

    Attributes
    ==========
    name: string
         name of the market environment
    pricing_date: datetime object
         date of the market environment

    Methods
    =======
    add_constant:
        adds a constant (e.g. model parameter)
    get_constant:
        gets a constant
    add_list:
        adds a list (e.g. underlyings)
    get_list:
        gets a list
    add_curve:
        adds a market curve (e.g. yield curve)
    get_curve:
        gets a market curve
    add_environment:
        adds and overwrites whole market environments with constants, lists, and curves
    """

    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        # overwrites existing values, if they exist
        self.constants.update(env.constants)
        self.lists.update(env.lists)
        self.curves.update(env.curves)


def sn_random_numbers(shape, antithetic=True, moment_matching=True, fixed_seed=True):
    """
    generate a ndarray of shape shape with pseudo-random numbers that are standard normally distributed

    :param shape: tuple(o,m,n)
           generation of array with shape (o,m,n)
    :param antithetic: Boolean
           generation of antithetic variables (look at Chapter 12 of "python for finance")
    :param moment_matching: Boolean
           matching of first and second moments (look at Chapter 12 of "python for finance")
    :param fixed_seed: Boolean
           flag ot fix the seed
    :return: (o,m,n) array of pseudo-random numbers
    """
    if fixed_seed:
        np.random.seed(1000)
    if antithetic:
        ran = np.random.standard_normal(
            (shape[0], shape[1], shape[2] // 2)
        )
        ran = np.concatenate((ran, -ran), axis=2)
    else:
        np.random.standard_normal(shape)
    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    if shape[0] == 1:
        return ran[0]
    else:
        return ran


class simulation_class(object):
    """
    Providing base methods for simulation classes.
    Attributes
    ==========
    name: str
        name of the object
    mar_env: instance of market_environment
       market environment data for simulation
    corr: bool
       True if correlated with other model object
    Methods
    =======
    generate_time_grid:
        returns time grid for simulation
    get_instrument_values:
        returns the current instrument values (array)
    """

    def __init__(self, name, mar_env, corr):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        self.initial_value = mar_env.get_constant('initial_value')
        self.volatility = mar_env.get_constant('volatility')
        self.final_date = mar_env.get_constant('final_date')
        self.currency = mar_env.get_constant('currency')
        self.frequency = mar_env.get_constant('frequency')
        self.paths = mar_env.get_constant('paths')
        self.discount_curve = mar_env.get_curve('discount_curve')
        try:
            # if time_grid in mar_env take that object
            # (for portfolio valuation)
            self.time_grid = mar_env.get_list('time_grid')
        except:
            self.time_grid = None
        try:
            # if there are special dates, then add these
            self.special_dates = mar_env.get_list('special_dates')
        except:
            self.special_dates = []
        self.instrument_values = None
        self.correlated = corr
        if corr is True:
            # only needed in a portfolio context when risk factors are correlated
            self.cholesky_matrix = mar_env.get_list('cholesky_matrix')
            self.rn_set = mar_env.get_list('rn_set')[self.name]
            self.random_numbers = mar_env.get_list('random_numbers')

    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date
        # pandas date_range function
        # freq = e.g. 'B' for Business Day,
        # 'W' for Weekly, 'M' for Monthly
        time_grid = pd.date_range(start=start, end=end, freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)
        # enhance time_grid by start, end, and special_dates
        if start not in time_grid:
            time_grid.insert(0, start)
            # insert start date if not in list
        if end not in time_grid:
            time_grid.append(end)

            # insert end date if not in list
        if len(self.special_dates) > 0:
            # add all special dates
            time_grid.extend(self.special_dates)
            # delete duplicates
            time_grid = list(set(time_grid))
            # sort list
            time_grid.sort()
        self.time_grid = np.array(time_grid)

    def get_instrument_values(self, fixed_seed=True):
        if self.instrument_values is None:
            # only initiate simulation if there are no instrument values
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        elif fixed_seed is False:
            # also initiate resimulation when fixed_seed is False
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.)
        return self.instrument_values


class geometric_brownian_motion(simulation_class):
    """ Class to generate simulated paths based on
    the Black-Scholes-Merton geometric Brownian motion model.
    Attributes
    ==========
    name: string
    name of the object
    mar_env: instance of market_environment
    market environment data for simulation
    corr: Boolean
    True if correlated with other model simulation object
     Methods
     =======
     update:
     updates parameters
     generate_paths:
     returns Monte Carlo paths given the market environment
     """

    def __init__(self, name, mar_env, corr=False):
        super(geometric_brownian_motion, self).__init__(name, mar_env, corr)

    def update(self, initial_value=None, volatility=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value

        if volatility is not None:
            self.volatility = volatility
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            # method from generic simulation class
            self.generate_time_grid()
        # number of dates for time grid
        M = len(self.time_grid)
        # number of paths
        I = self.paths
        # ndarray initialization for path simulation
        paths = np.zeros((M, I))
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if not self.correlated:
            # if not correlated, generate random numbers
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)

        else:
            # if correlated, use random number object as provided in market environment
            rand = self.random_numbers
        short_rate = self.discount_curve.short_rate
        # get short rate for drift of process
        for t in range(1, len(self.time_grid)):
            # select the right time slice from the relevant
            # random number set
            if not self.correlated:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            # difference between two dates as year fraction
            paths[t] = paths[t - 1] * np.exp((short_rate - 0.5 *
                                              self.volatility ** 2) * dt +
                                             self.volatility * np.sqrt(dt) * ran)
            # generate simulated values for the respective date
        self.instrument_values = paths


class jump_diffusion(simulation_class):
    """ Class to generate simulated paths based on
    the Merton (1976) jump diffusion model.
    Attributes
    ==========
    name: str
    name of the object
    mar_env: instance of market_environment
    market environment data for simulation
    corr: bool
    True if correlated with other model object
    Methods
    =======
    update:
    updates parameters
    generate_paths:
    returns Monte Carlo paths given the market environment
    """

    def __init__(self, name, mar_env, corr=False):
        super(jump_diffusion, self).__init__(name, mar_env, corr)
        # additional parameters needed
        self.lamb = mar_env.get_constant('lambda')
        self.mu = mar_env.get_constant('mu')
        self.delt = mar_env.get_constant('delta')

    def update(self, initial_value=None, volatility=None, lamb=None, mu=None, delta=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delt = delta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            # method from generic simulation class
            self.generate_time_grid()
        # number of dates for time grid
        M = len(self.time_grid)
        # number of paths
        I = self.paths
        # ndarray initialization for path simulation
        paths = np.zeros((M, I))
        # initialize first date with initial_value
        paths[0] = self.initial_value
        if self.correlated is False:
            # if not correlated, generate random numbers
            sn1 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            # if correlated, use random number object as provided in market environment
            sn1 = self.random_numbers
        # standard normally distributed pseudo-random numbers
        # for the jump component
        sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt ** 2) - 1)
        short_rate = self.discount_curve.short_rate
        for t in range(1, len(self.time_grid)):
            if self.correlated is False:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            # difference between two dates as year fraction
            poi = np.random.poisson(self.lamb * dt, I)
            # Poisson-distributed pseudo-random numbers for jump component
            paths[t] = paths[t - 1] * (np.exp(
                (short_rate - rj - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * ran) + (
                                               np.exp(self.mu + self.delt * sn2[t]) - 1) * poi)
        self.instrument_values = paths


class square_root_diffusion(simulation_class):
    """ Class to generate simulated paths based on
    the Cox-Ingersoll-Ross (1985) square-root diffusion model.
    Attributes
    ==========
    name : string
    name of the object
    mar_env : instance of market_environment
    market environment data for simulation
    corr : Boolean
    True if correlated with other model object
    Methods
    =======
    update :
    updates parameters
    generate_paths :
    returns Monte Carlo paths given the market environment
    """

    def __init__(self, name, mar_env, corr=False):
        super(square_root_diffusion, self).__init__(name, mar_env, corr)
        # additional parameters needed
        self.kappa = mar_env.get_constant('kappa')
        self.theta = mar_env.get_constant('theta')

    def update(self, initial_value=None, volatility=None, kappa=None, theta=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=True, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        paths_[0] = self.initial_value
        if self.correlated is False:
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            rand = self.random_numbers
        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            # full truncation Euler discretization
            paths_[t] = (paths_[t - 1] + self.kappa *
                         (self.theta - np.maximum(0, paths_[t - 1, :])) * dt +
                         np.sqrt(np.maximum(0, paths_[t - 1, :])) *
                         self.volatility * np.sqrt(dt) * ran)
            paths[t] = np.maximum(0, paths_[t])
            self.instrument_values = paths


class stochatic_volatility(simulation_class):
    def __init__(self, name, mar_env,corr=False):
        super(stochatic_volatility, self).__init__(name, mar_env,corr)
        # additional parameters needed
        self.kappa = mar_env.get_constant('kappa')
        self.theta = mar_env.get_constant('theta')
        self.initial_vol = mar_env.get_constant('initial_volatility')
        self.volatility_vol = mar_env.get_constant("volatility_vol")
        self.rho = mar_env.get_constant("rho")

        self.cholesky_matrix = np.array(((1,self.rho),(self.rho,1)))


    def generate_paths(self,fixed_seed=True, day_count=365.):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths

        run = sn_random_numbers((2,M,I))
        vh = np.zeros_like(run[0])
        v = np.zeros_like(run[0])
        v[0] = self.initial_vol
        vh[0] = self.initial_vol
        short_rate = self.discount_curve.short_rate
        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            ran = np.dot(self.cholesky_matrix, run[:, t, :])

            # full truncation Euler discretization
            vh[t,:] = (vh[t - 1] + self.kappa *
                         (self.theta - np.maximum(0, vh[t - 1,:])) * dt +
                         np.sqrt(np.maximum(0, vh[t - 1, :])) *
                         self.volatility_vol * np.sqrt(dt) * ran[1])
            v[t,:] = np.maximum(0, vh[t])

        s = np.zeros_like(v)
        s[0,:] = self.initial_value

        for t in range(1,len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            ran = np.dot(self.cholesky_matrix, run[:, t, :])
            s[t,:] = s[t-1]*np.exp((short_rate - 0.5 * v[t]) * dt + np.sqrt(v[t]) * ran[0] * np.sqrt(dt))
        plt.plot(s)
        plt.show()

        self.instrument_values = s

class valuation_class(object):
    """ Basic class for single-factor valuation.
    Attributes
    ==========
    name: str
    name of the object
    underlying: instance of simulation class
    object modeling the single risk factor
    mar_env: instance of market_environment
    market environment data for valuation
    payoff_func: str
    derivatives payoff in Python syntax
    Example: 'np.maximum(maturity_value - 100, 0)'
    where maturity_value is the NumPy vector with
    respective values of the underlying
    Example: 'np.maximum(instrument_values - 100, 0)'
    where instrument_values is the NumPy matrix with
    values of the underlying over the whole time/path grid
    Methods
    =======
    update:
    updates selected valuation parameters
    delta:
    returns the delta of the derivative
    vega:
    returns the vega of the derivative
    """

    def __init__(self, name, underlying, mar_env, payoff_func=''):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        try:
            # strike is optional
            self.strike = mar_env.get_constant('strike')
        except:
            pass
        self.maturity = mar_env.get_constant('maturity')
        self.currency = mar_env.get_constant('currency')
        # simulation parameters and discount curve from simulation object
        self.frequency = underlying.frequency
        self.paths = underlying.paths
        self.discount_curve = underlying.discount_curve
        self.payoff_func = payoff_func
        self.underlying = underlying
        # provide pricing_date and maturity to underlying
        self.underlying.special_dates.extend([self.pricing_date, self.maturity])

    def update(self, initial_value=None, volatility=None, strike=None, maturity=None):
        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)

        if volatility is not None:
            self.underlying.update(volatility=volatility)
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            # add new maturity date if not in time_grid
            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None

    def delta(self, interval=None, accuracy=4):
        if interval is None:
            interval = self.underlying.initial_value / 50.

        # forward difference approximation
        # calculate left value for numerical delta
        value_left = self.present_value(fixed_seed=True)
        # numerical underlying value for right value
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value=initial_del)
        # calculate right value for numerical delta
        value_right = self.present_value(fixed_seed=True)
        # reset the initial_value of the simulation object
        self.underlying.update(initial_value=initial_del - interval)
        delta = (value_right - value_left) / interval
        # correct for potential numerical errors
        if delta < -1.0:
            return -1.0
        elif delta > 1.0:
            return 1.0
        else:
            return round(delta, accuracy)

    def vega(self, interval=0.01, accuracy=4):
        if interval < self.underlying.volatility / 50.:
            interval = self.underlying.volatility / 50.

        # forward-difference approximation
        # calculate the left value for numerical vega
        value_left = self.present_value(fixed_seed=True)
        # numerical volatility value for right value
        vola_del = self.underlying.volatility + interval
        # update the simulation object
        self.underlying.update(volatility=vola_del)
        # calculate the right value for numerical vega
        value_right = self.present_value(fixed_seed=True)
        # reset volatility value of simulation object
        self.underlying.update(volatility=vola_del - interval)
        vega = (value_right - value_left) / interval
        return round(vega, accuracy)


class valuation_mcs_european(valuation_class):
    """ Class to value European options with arbitrary payoff by single-factor Monte Carlo simulation.
    Methods
    =======
    generate_payoff:
    returns payoffs given the paths and the payoff function
    present_value:
    returns present value (Monte Carlo estimator)
    """

    def generate_payoff(self, fixed_seed=False):
        """
        Parameters
        ==========
        fixed_seed: bool
        use same/fixed seed for valuation
        """
        try:
            # strike is optional
            strike = self.strike
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index)
        except:
            print('Maturity date not in time grid of underlying.')
        maturity_value = paths[time_index]
        # average value over whole path
        mean_value = np.mean(paths[:time_index], axis=1)
        # maximum value over whole path
        max_value = np.amax(paths[:time_index], axis=1)[-1]
        # minimum value over whole path
        min_value = np.amin(paths[:time_index], axis=1)[-1]
        try:
            payoff = eval(self.payoff_func)
            return payoff
        except:
            print('Error evaluating payoff function.')

    def present_value(self, accuracy=6, fixed_seed=False, full=False):
        """
        Parameters
        ==========
        accuracy: int
             number of decimals in returned result
        fixed_seed: bool
             use same/fixed seed for valuation
        full: bool
        return also full 1d array of present values
        """
        cash_flow = self.generate_payoff(fixed_seed=fixed_seed)
        discount_factor = self.discount_curve.get_discount_factors(
            (self.pricing_date, self.maturity))[0, 1]
        result = discount_factor * np.sum(cash_flow) / len(cash_flow)
        if full:
            return round(result, accuracy), discount_factor * cash_flow
        else:
            return round(result, accuracy)


def plot_option_stats(s_list, p_list, d_list, v_list):
    """Plots option prices, deltas, and vegas for a set of
    different initial values of the underlying.
    Parameters
    ==========
    s_list: array or list
    set of initial values of the underlying
    p_list: array or list
    present values
    European Exercise | 605
    d_list: array or list
    results for deltas
    v_list: array or list
    results for vegas
    """
    plt.figure(figsize=(10, 7))
    sub1 = plt.subplot(311)
    plt.plot(s_list, p_list, 'ro', label='present value')
    plt.plot(s_list, p_list, 'b')
    plt.legend(loc=0)
    plt.setp(sub1.get_xticklabels(), visible=False)
    sub2 = plt.subplot(312)
    plt.plot(s_list, d_list, 'go', label='Delta')
    plt.plot(s_list, d_list, 'b')
    plt.legend(loc=0)
    plt.ylim(min(d_list) - 0.1, max(d_list) + 0.1)
    plt.setp(sub2.get_xticklabels(), visible=False)
    sub3 = plt.subplot(313)
    plt.plot(s_list, v_list, 'yo', label='Vega')
    plt.plot(s_list, v_list, 'b')
    plt.xlabel('initial value of underlying')
    plt.legend(loc=0)

class valuation_mcs_american(valuation_class):
    """ Class to value American options with arbitrary payoff
    by single-factor Monte Carlo simulation.
    Methods
    =======
    generate_payoff:
    returns payoffs given the paths and the payoff function
    present_value:
    returns present value (LSM Monte Carlo estimator)
    according to Longstaff-Schwartz (2001)
    """
    def generate_payoff(self, fixed_seed=False):
        """
        Parameters
        ==========
        fixed_seed:
        use same/fixed seed for valuation
        """
        try:
            # strike is optional
            strike = self.strike
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        time_index_start = int(np.where(time_grid == self.pricing_date)[0])
        time_index_end = int(np.where(time_grid == self.maturity)[0])
        instrument_values = paths[time_index_start:time_index_end + 1]
        payoff = eval(self.payoff_func)
        return instrument_values, payoff, time_index_start, time_index_end

    def present_value(self, accuracy=6, fixed_seed=False, bf=5, full=False):
        """
        Parameters
        ==========
        accuracy: int
        number of decimals in returned result
        fixed_seed: bool
        use same/fixed seed for valuation
        bf: int
        number of basis functions for regression
        full: bool
        return also full 1d array of present values
        """
        instrument_values, inner_values, time_index_start, time_index_end = \
            self.generate_payoff(fixed_seed=fixed_seed)
        time_list = self.underlying.time_grid[
                    time_index_start:time_index_end + 1]
        discount_factors = self.discount_curve.get_discount_factors(
            time_list, dtobjects=True)
        V = inner_values[-1]
        for t in range(len(time_list) - 2, 0, -1):
            # derive relevant discount factor for given time interval
            df = discount_factors[t, 1] / discount_factors[t + 1, 1]
            # regression step
            rg = np.polyfit(instrument_values[t], V * df, bf)
            # calculation of continuation values per path
            C = np.polyval(rg, instrument_values[t])
            # optimal decision step:
            # if condition is satisfied (inner value > regressed cont. value)
            # then take inner value; take actual cont. value otherwise
            V = np.where(inner_values[t] > C, inner_values[t], V * df)
        df = discount_factors[0, 1] / discount_factors[1, 1]
        result = df * np.sum(V) / len(V)
        if full:
            return round(result, accuracy), df * V
        else:
            return round(result, accuracy)

