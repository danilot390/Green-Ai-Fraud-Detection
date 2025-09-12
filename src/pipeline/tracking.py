from codecarbon import EmissionsTracker

def start_tracker(experiment_config, experiment_dir, logger):
    """
    Starts energy consumption tracker.
    """
    tracker_on = experiment_config['tracker'].get('enabled', False)

    # --- Initialize CodeCarbon Traker ---
    if not tracker_on:
        logger.warning("Energy consumption tracking is disabled in the configuration.")
        return None
    else:
        if experiment_config['tracker'].get('type', 'CodeCarbon') == 'CodeCarbon':
            tracker = EmissionsTracker(
                project_name=experiment_config['tracker'].get('project', 'Fraud_Detection'),
                output_dir=experiment_dir, 
                log_level='error',
                )
            tracker.start()
            logger.info(f"Measuring energy consumption with {experiment_config['tracker'].get('type', 'CodeCarbon') }...")
            return tracker

def stop_tracker(tracker, logger):
    """
    Ends energy consumption tracker.
    """
    if not tracker == None:
        tracker.stop()
        logger.info('Tracker stopped.')

        return tracker.final_emissions
    
    return tracker
