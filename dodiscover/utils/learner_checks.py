def check_learner(estimator=None, generate_only=False, Estimator="deprecated"):
    """Check if learner adheres to dodiscover conventions.

    This function will run an extensive test-suite for input validation,
    shapes, etc, making sure that the estimator complies with `dodiscover`
    conventions as detailed in :ref:`rolling_your_own_learner`.

    Setting `generate_only=True` returns a generator that yields (estimator,
    check) tuples where the check can be called independently from each
    other, i.e. `check(estimator)`. This allows all checks to be run
    independently and report the checks that are failing.

    dodiscover provides a pytest specific decorator,
    :func:`~dodiscover.utils.parametrize_with_checks`, making it easier to test
    multiple estimators.

    Parameters
    ----------
    learner : learner object
        Learner instance to check.

    generate_only : bool, default=False
        When `False`, checks are evaluated when `check_estimator` is called.
        When `True`, `check_estimator` returns a generator that yields
        (estimator, check) tuples. The check is run by calling
        `check(estimator)`.

        .. versionadded:: 0.22

    Estimator : estimator object
        Estimator instance to check.

        .. deprecated:: 1.1
            ``Estimator`` was deprecated in favor of ``estimator`` in version 1.1
            and will be removed in version 1.3.

    Returns
    -------
    checks_generator : generator
        Generator that yields (estimator, check) tuples. Returned when
        `generate_only=True`.

    See Also
    --------
    parametrize_with_checks : Pytest specific decorator for parametrizing estimator
        checks.
    """

    if estimator is None and Estimator == "deprecated":
        msg = "Either estimator or Estimator should be passed to check_estimator."
        raise ValueError(msg)

    if Estimator != "deprecated":
        msg = (
            "'Estimator' was deprecated in favor of 'estimator' in version 1.1 "
            "and will be removed in version 1.3."
        )
        warnings.warn(msg, FutureWarning)
        estimator = Estimator
    if isinstance(estimator, type):
        msg = (
            "Passing a class was deprecated in version 0.23 "
            "and isn't supported anymore from 0.24."
            "Please pass an instance instead."
        )
        raise TypeError(msg)

    name = type(estimator).__name__

    def checks_generator():
        for check in _yield_all_checks(estimator):
            check = _maybe_skip(estimator, check)
            yield estimator, partial(check, name)

    if generate_only:
        return checks_generator()

    for estimator, check in checks_generator():
        try:
            check(estimator)
        except SkipTest as exception:
            # SkipTest is thrown when pandas can't be imported, or by checks
            # that are in the xfail_checks tag
            warnings.warn(str(exception), SkipTestWarning)
