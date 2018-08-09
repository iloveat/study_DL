# -*- coding: utf-8 -*-
import os
import subprocess


sptk_dir = '/home/brycezou/works/tuling_tts_train/tools/bin/SPTK-3.9/'
world_dir = '/home/brycezou/works/tuling_tts_train/tools/bin/WORLD/'

SPTK = {
    'X2X'    : os.path.join(sptk_dir, 'x2x'),
    'MERGE'  : os.path.join(sptk_dir, 'merge'),
    'BCP'    : os.path.join(sptk_dir, 'bcp'),
    'MLPG'   : os.path.join(sptk_dir, 'mlpg'),
    'MGC2SP' : os.path.join(sptk_dir, 'mgc2sp'),
    'VSUM'   : os.path.join(sptk_dir, 'vsum'),
    'VSTAT'  : os.path.join(sptk_dir, 'vstat'),
    'SOPR'   : os.path.join(sptk_dir, 'sopr'),
    'VOPR'   : os.path.join(sptk_dir, 'vopr'),
    'FREQT'  : os.path.join(sptk_dir, 'freqt'),
    'C2ACR'  : os.path.join(sptk_dir, 'c2acr'),
    'MC2B'   : os.path.join(sptk_dir, 'mc2b'),
    'B2MC'   : os.path.join(sptk_dir, 'b2mc')
}

WORLD = {
    'SYNTHESIS'     : os.path.join(world_dir, 'synth'),
    'ANALYSIS'      : os.path.join(world_dir, 'analysis'),
}


def run_process(args):
    try:
        # the following is only available in later versions of Python
        # rval = subprocess.check_output(args)

        # bufsize=-1 enables buffering and may improve performance compared to the unbuffered case
        p = subprocess.Popen(args, bufsize=-1, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, close_fds=True)
        # better to use communicate() than read() and write() - this avoids deadlocks
        std_out_data, std_err_data = p.communicate()

        if p.returncode != 0:
            # for critical things, we always log, even if log==False
            print 'exit status: %d' % p.returncode
            print 'for command: %s' % args
            print '     stderr: %s' % std_err_data
            print '     stdout: %s' % std_out_data
            raise OSError

        return std_out_data, std_err_data

    except subprocess.CalledProcessError as e:
        # not sure under what circumstances this exception would be raised in Python 2.6
        print 'exit status %d' % e.returncode
        print ' for command: %s' % args
        # not sure if there is an 'output' attribute under 2.6 ? still need to test this...
        print '      output: %s' % e.output
        raise

    except ValueError:
        print 'ValueError for %s' % args
        raise

    except OSError:
        print 'OSError for %s' % args
        raise

    except KeyboardInterrupt:
        print 'KeyboardInterrupt during %s' % args
        try:
            # try to kill the subprocess, if it exists
            p.kill()
        except UnboundLocalError:
            # this means that p was undefined at the moment of the keyboard interrupt
            # (and we do nothing)
            pass
        raise KeyboardInterrupt


def generate_wave(output_dir, file_id_list):
    pf_coef = 1.4
    fw_coef = 0.58
    co_coef = 511
    fl_coef = 1024
    fw_alpha = 0.58
    fl = 1024
    sr = 16000
    mgc_dim = 60
    do_post_filtering = True

    for filename in file_id_list:
        base = filename
        files = {'ap': base + '.ap',
                 'sp': base + '.sp',
                 'f0': base + '.f0',
                 'mgc': base + '.mgc',
                 'lf0': base + '.lf0',
                 'bap': base + '.bap',
                 'wav': base + '.wav'}

        mgc_file_name = files['mgc']
        curr_dir = os.getcwd()
        os.chdir(output_dir)

        # post-filtering
        if do_post_filtering:
            line = "echo 1 1 "
            for i in range(2, mgc_dim):
                line = line + str(pf_coef) + " "

            run_process('{line} | {x2x} +af > {weight}'
                        .format(line=line, x2x=SPTK['X2X'], weight=os.path.join(output_dir, 'weight')))

            run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
                        .format(freqt=SPTK['FREQT'], order=mgc_dim-1, fw=fw_coef, co=co_coef, mgc=files['mgc'],
                                c2acr=SPTK['C2ACR'], fl=fl_coef, base_r0=files['mgc']+'_r0'))

            run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
                        .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=files['mgc'],
                                weight=os.path.join(output_dir, 'weight'), freqt=SPTK['FREQT'], fw=fw_coef, co=co_coef,
                                c2acr=SPTK['C2ACR'], fl=fl_coef, base_p_r0=files['mgc']+'_p_r0'))

            run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'
                        .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=files['mgc'],
                                weight=os.path.join(output_dir, 'weight'), mc2b=SPTK['MC2B'], fw=fw_coef, bcp=SPTK['BCP'],
                                base_b0=files['mgc']+'_b0'))

            run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                        .format(vopr=SPTK['VOPR'], base_r0=files['mgc']+'_r0', base_p_r0=files['mgc']+'_p_r0',
                                sopr=SPTK['SOPR'], base_b0=files['mgc']+'_b0', base_p_b0=files['mgc']+'_p_b0'))

            run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                        .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=files['mgc'],
                                weight=os.path.join(output_dir, 'weight'), mc2b=SPTK['MC2B'], fw=fw_coef, bcp=SPTK['BCP'],
                                merge=SPTK['MERGE'], order2=mgc_dim-2, base_p_b0=files['mgc']+'_p_b0',
                                b2mc=SPTK['B2MC'], base_p_mgc=files['mgc']+'_p_mgc'))

            mgc_file_name = files['mgc']+'_p_mgc'

        # synthesize audio
        run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'
                    .format(sopr=SPTK['SOPR'], lf0=files['lf0'], x2x=SPTK['X2X'], f0=files['f0']))

        run_process('{sopr} -c 0 {bap} | {x2x} +fd > {ap}'
                    .format(sopr=SPTK['SOPR'], bap=files['bap'], x2x=SPTK['X2X'], ap=files['ap']))

        # If using world v2, please comment above line and uncomment this
        # run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 0 {bap} | {sopr} -d 32768.0 -P | {x2x} +fd > {ap}'
        #            .format(mgc2sp=SPTK['MGC2SP'], alpha=fw_alpha, order=bap_dim, fl=fl, bap=bap_file_name,
        #                    sopr=SPTK['SOPR'], x2x=SPTK['X2X'], ap=files['ap']))

        run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
                    .format(mgc2sp=SPTK['MGC2SP'], alpha=fw_alpha, order=mgc_dim-1, fl=fl,
                            mgc=mgc_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], sp=files['sp']))

        run_process('{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'
                    .format(synworld=WORLD['SYNTHESIS'], fl=fl, sr=sr, f0=files['f0'], sp=files['sp'],
                            ap=files['ap'], wav=files['wav']))

        run_process('rm -f {ap} {sp} {f0}'.format(ap=files['ap'], sp=files['sp'], f0=files['f0']))

        os.chdir(curr_dir)

        print 'wrote to wave %s/synthesis/%s.wav' % (curr_dir, filename)













