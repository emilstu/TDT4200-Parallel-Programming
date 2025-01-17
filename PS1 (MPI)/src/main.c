#include <crypt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>

#include "util.h"

static bool run_master(int argc, char **argv, int mpi_size);
static bool run_replica(int mpi_size);
static bool init_master(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv, int mpi_size);
static void master_cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict);
static void master_crack(ExtendedCrackResult *result, Options *options, Dictionary *dict, ShadowEntry *entry, int mpi_size);
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict);
static void crack_job(CrackResult *result, CrackJob *job);
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file);
static void handle_overview_result(Options *options, OverviewCrackResult *overview_result);


/*
 * Main entrypoint.
 */
int main(int argc, char **argv) {
    int mpi_size;
    int my_rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    bool success;
    
    if (my_rank == 0) {
        success = run_master(argc, argv, mpi_size);
    }else{
        success = run_replica(mpi_size);
    }

    MPI_Finalize();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}


static bool run_replica(int mpi_size) {
    bool work = true;

    while (work) {
        CrackJob rep_job;
        CrackResult rep_result;

        MPI_Scatter(0, 0, MPI_BYTE, &rep_job, sizeof(CrackJob), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rep_job.action == ACTION_STOP) {
            work = false;
        
        }else if (rep_job.action == ACTION_WAIT) {
            rep_result.status = STATUS_SKIP;
        
        }else if (rep_job.action == ACTION_WORK) {
            crack_job(&rep_result, &rep_job);
        }

        MPI_Gather(&rep_result, sizeof(CrackResult), MPI_BYTE, 0, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    return true;
}


/*
 * Entrypoint for master.
 */
static bool run_master(int argc, char **argv, int mpi_size) {    
    Options options = {};
    Dictionary dict = {};
    FILE *shadow_file = NULL;
    FILE *result_file = NULL;
    OverviewCrackResult overview_result = {};

    bool init_success = init_master(&options, &dict, &shadow_file, &result_file, argc, argv, mpi_size);

    // If init successful, try to crack all shadow entries
    if (!options.quiet) {
        printf("\nEntries:\n");
    }

    if (init_success) {
        ShadowEntry shadow_entry;
        while (get_next_shadow_entry(&shadow_entry, shadow_file)) {
            ExtendedCrackResult result;
            master_crack(&result, &options, &dict, &shadow_entry, mpi_size);
            handle_result(&options, &result, &overview_result, result_file);
        }
    }

    //Stop all replicas by sending jobs with ACTION_STOP  
    CrackJob *jobs = calloc(mpi_size, sizeof(CrackJob));

    for (int i = 0; i < mpi_size; i++) {
        jobs[i].action = ACTION_STOP;
    }

    CrackJob stop_job;
    MPI_Scatter(jobs, sizeof(CrackJob), MPI_BYTE, &stop_job, sizeof(CrackJob), MPI_BYTE, 0, MPI_COMM_WORLD);   
       
    handle_overview_result(&options, &overview_result);
    master_cleanup(shadow_file, result_file, &dict);

    free(jobs);

    return true;
}

/*
 * Initialize master stuff.
 */
static bool init_master(Options *options, Dictionary *dict, FILE **shadow_file, FILE **result_file, int argc, char **argv, int mpi_size) {

 
    // Parse CLI args
    if (!parse_cli_args(options, argc, argv)) {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }

    // Print some useful info
    if (!options->quiet) {
        // TODO
        printf("Workers: %d\n", mpi_size);
        printf("Max symbols: %ld\n", options->max_length);
        printf("Symbol separator: \"%s\"\n", options->separator);
    }

    // Open shadow file
    if (!options->quiet) {
        printf("Shadow file: %s\n", options->shadow_file);
    }
    if (!open_file(shadow_file, options->shadow_file, "r")) {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }
    // Open output file if provided
    if (options->result_file[0] != 0) {
        if (!options->quiet) {
            printf("Output file: %s\n", options->result_file);
        }
        if (!open_file(result_file, options->result_file, "w")) {
            master_cleanup(*shadow_file, *result_file, dict);
            return false;
        }
    }
    // Read full directory
    if (!options->quiet) {
        printf("Dictionary file: %s\n", options->dict_file);
    }
    if (!read_dictionary(dict, options, options->dict_file)) {
        master_cleanup(*shadow_file, *result_file, dict);
        return false;
    }
    
    return true;

}

/*
 * Cleanup master stuff.
 */
static void master_cleanup(FILE *shadow_file, FILE *result_file, Dictionary *dict) {
    if (shadow_file) {
        fclose(shadow_file);
    }
    if (result_file) {
        fclose(result_file);
    }
    if (dict->elements) {
        free(dict->elements);
    }
}

/*
 * Crack a shadow password entry as master.
 */
static void master_crack(ExtendedCrackResult *result, Options *options, Dictionary *dict, ShadowEntry *entry, int mpi_size) {
    // Initialize result
    memset(result, 0, sizeof(ExtendedCrackResult));
    strncpy(result->user, entry->user, MAX_SHADOW_USER_LENGTH);
    strncpy(result->passfield, entry->passfield, MAX_SHADOW_PASSFIELD_LENGTH);
    result->alg = entry->alg;

    // Accept only known algs
    if (entry->alg == ALG_UNKNOWN) {
        result->status = STATUS_SKIP;
        return;
    }

    // Setup vars for cracking
    ProbeConfig config = {};
    config.dict_positions = calloc(options->max_length, sizeof(size_t));
    config.symbols = calloc(options->max_length, MAX_DICT_ELEMENT_LENGTH + 1);
    
    // TODO allocate and initialize for multiple ranks
    CrackJob *jobs = calloc(mpi_size, sizeof(CrackJob));
    CrackResult *results = calloc(mpi_size, sizeof(CrackResult)); 

    for (int i = 0; i < mpi_size; i++) {
        strncpy(jobs[i].passfield, entry->passfield, MAX_SHADOW_PASSFIELD_LENGTH);
    }
    
    // Start time measurement
    // TODO
    double start_time = MPI_Wtime();

    // Try probes until the status changes (when a match is found or the search space is exhausted)
    while(result->status == STATUS_PENDING) {
        
        // Make jobs with new probes
        // TODO for all ranks
        // TODO use ACTION_WAIT for jobs when running out of probes
        
        for (int i = 0; i < mpi_size; i++) { 
            memset(&jobs[i], 0, sizeof(CrackJob));
            memset(&results[0], 0, sizeof(CrackResult));

            bool more_probes = get_next_probe(&config, options, dict);
            if (!more_probes) {
                jobs[i].action = ACTION_WAIT;
                break;
            }
                
            jobs[i].action = ACTION_WORK;
            strncpy(jobs[i].passfield, entry->passfield, MAX_PASSWORD_LENGTH);
            jobs[i].alg = entry->alg;
            jobs[i].salt_end = entry->salt_end;
            strncpy(jobs[i].probe, config.probe, MAX_PASSWORD_LENGTH);
            if (options->verbose) {
                printf("%s\n", jobs[i].probe);
            }
        }
        
        // Process job
        // TODO parallelize
        CrackJob master_job;
        CrackResult master_result;

        MPI_Scatter(jobs, sizeof(CrackJob), MPI_BYTE, &master_job, sizeof(CrackJob), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        if (master_job.action == ACTION_WORK) {
            crack_job(&master_result, &master_job);
        } else {
            master_result.status = STATUS_SKIP;
        }

        MPI_Gather(&master_result, sizeof(CrackResult), MPI_BYTE, results, sizeof(CrackResult), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Handle results
        // TODO for all ranks
        
        result->attempts++;
        //printf("Attempts: %ld\n", result->attempts);

        for (int i = 0; i < mpi_size; i ++){
            // Accept if success (currently the only one it makes sense to stop on)
            //printf("Status: %s\n", crack_result_status_to_string(results[i].status));
            if (results[i].status != STATUS_PENDING) {
                result->status = results[i].status;
                printf("Status: %s\n", crack_result_status_to_string(results[i].status));
                strncpy(result->password, results[i].password, MAX_PASSWORD_LENGTH);
            }
        }
        
    }

    // End time measurement
    // TODO
    double end_time = MPI_Wtime();
    result->duration = end_time - start_time;

    free(config.dict_positions);
    free(config.symbols);
    free(jobs);
    free(results);
}

/*
 * Build the next probe. Returns false with an empty probe when the search space is exhausted.
 */
static bool get_next_probe(ProbeConfig *config, Options *options, Dictionary *dict) {
    // Check if dict is empty
    if (dict->length == 0) {
        return false;
    }

    // Find last symbol which can be replaced with the next one, if any exists
    ssize_t last_replaceable_pos = -1;
    for (size_t i = 0; i < config->size; i++) {
        if (config->dict_positions[i] < dict->length - 1) {
            last_replaceable_pos = i;
        }
    }

    // A symbol can be replaced, replace last one and reset all behind it
    if (last_replaceable_pos >= 0) {
        size_t new_dict_pos = config->dict_positions[last_replaceable_pos] + 1;
        config->dict_positions[last_replaceable_pos] = new_dict_pos;
        strncpy(config->symbols[last_replaceable_pos], dict->elements[new_dict_pos], MAX_DICT_ELEMENT_LENGTH);
        for (size_t i = last_replaceable_pos + 1; i < config->size; i++) {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }
    // No symbols can be replaced and no more symbols are allowed, return error
    else if (config->size == options->max_length) {
        config->probe[0] = 0;
        return false;
    }
    // New symbol can be added, reset all previous positions and add it
    else {
        config->size++;
        for (size_t i = 0; i < config->size; i++) {
            config->dict_positions[i] = 0;
            strncpy(config->symbols[i], dict->elements[0], MAX_DICT_ELEMENT_LENGTH);
        }
    }

    // Build probe
    config->probe[0] = 0;
    for (size_t i = 0; i < config->size; i++) {
        if (i > 0) {
            strncat(config->probe, options->separator, MAX_PASSWORD_LENGTH);
        }
        strncat(config->probe, config->symbols[i], MAX_PASSWORD_LENGTH);
    }

    return true;
}

/*
 * Handle result from trying to crack a single password.
 */
static void handle_result(Options *options, ExtendedCrackResult *result, OverviewCrackResult *overview_result, FILE *result_file) {
    // Make representations
    char const *alg_str = cryptalg_to_string(result->alg);
    char const *status_str = crack_result_status_to_string(result->status);
    double attempts_per_second = result->attempts / result->duration;

    // Format and print
    size_t const static max_output_length = 1023;
    char *output = malloc(max_output_length + 1);
    snprintf(output, max_output_length + 1, "user=\"%s\" alg=\"%s\" status=\"%s\" duration=\"%fs\" attempts=\"%ld\" attempts_per_second=\"%f\" password=\"%s\"",
            result->user, alg_str, status_str, result->duration, result->attempts, attempts_per_second, result->password);
    if (!options->quiet) {
        printf("%s\n", output);
    }
    if (result_file) {
        fprintf(result_file, "%s\n", output);
        fflush(result_file);
    }
    free(output);

    // Update overview
    overview_result->statuses[result->status]++;
    overview_result->duration += result->duration;
    overview_result->attempts += result->attempts;
}

/*
 * Handle result from trying to crack all passwords.
 */
static void handle_overview_result(Options *options, OverviewCrackResult *result) {
    if (!options->quiet) {
        printf("\nOverview:\n");
        printf("Total duration: %.3fs\n", result->duration);
        printf("Total attempts: %ld\n", result->attempts);
        printf("Total attempts per second: %.3f\n", result->attempts / result->duration);
        printf("Skipped: %ld\n", result->statuses[STATUS_SKIP]);
        printf("Successful: %ld\n", result->statuses[STATUS_SUCCESS]);
        printf("Failed: %ld\n", result->statuses[STATUS_FAIL]);
    }
}

/*
 * Hash probe and compare.
 */
static void crack_job(CrackResult *result, CrackJob *job) {
    memset(result, 0, sizeof(CrackResult));

    // Only accept known (redundant check)
    if (job->alg == ALG_UNKNOWN) {
        result->status = STATUS_SKIP;
        return;
    }

    char const *new_passfield = crypt(job->probe, job->passfield);
    if (new_passfield != NULL && strncmp(job->passfield, new_passfield, MAX_SHADOW_PASSFIELD_LENGTH) == 0) {
        // Match found, abort search
        result->status = STATUS_SUCCESS;
        strncpy(result->password, job->probe, MAX_PASSWORD_LENGTH);
        result->password[MAX_PASSWORD_LENGTH] = 0;
    }
}
