
import os
import re
import shutil
from pyrouge import Rouge155

class Rouge155_Better(Rouge155):

    def cleanup(self, conf_file, remove=True):
        text = open(conf_file).read()
        reg = '<MODEL-ROOT>(.*?)</MODEL-ROOT>'
        m = re.search(reg, text)
        model_root = m.group(1)
        reg = '<PEER-ROOT>(.*?)</PEER-ROOT>'
        m = re.search(reg, text)
        peer_root = m.group(1)
        wast_dir1 = os.path.dirname(model_root)
        wast_dir2 = os.path.dirname(peer_root)
        assert wast_dir1 == wast_dir2, 'MUST be the same'
        conf_dir = os.path.dirname(conf_file)
        if remove:
            shutil.rmtree(wast_dir1)
            shutil.rmtree(conf_dir)

    def convert_and_evaluate_and_delete(self, system_id=1,
                                        split_sentences=False,
                                        rouge_args=None):
        output = self.convert_and_evaluate(system_id,
                                           split_sentences,
                                           rouge_args)
        self.cleanup(self.config_file, True)

        return output


def get_rouge(sysdir, refdir, cmd='-a -c 95 -m -n 4 -w 1.2', length=-1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(dir_path, 'ROUGE-RELEASE-1.5.5')
    if length != -1:
        cmd += ' -b %d' % length
    cmd += ' -e ' + script_path + '/data'
    r = Rouge155_Better(script_path)
    r.system_dir = sysdir
    r.model_dir = refdir
    r.system_filename_pattern = '(\d+).test'
    r.model_filename_pattern = '#ID#.gold'
    output = r.convert_and_evaluate_and_delete(rouge_args=cmd)
    output_dict = r.output_to_dict(output)

    return output_dict, output


def get_rouge_multi_ref(sysdir, refdir, cmd='-a -c 95 -m -n 4 -w 1.2', length=-1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(dir_path, 'ROUGE-RELEASE-1.5.5')
    if length != -1:
        cmd += ' -b %d' % length
    cmd += ' -e ' + script_path + '/data'
    r = Rouge155_Better(script_path)
    r.system_dir = sysdir
    r.model_dir = refdir
    r.system_filename_pattern = '(\d+).test'
    r.model_filename_pattern = '[A-Z].#ID#.gold'
    output = r.convert_and_evaluate_and_delete(rouge_args=cmd)
    # output = r.convert_and_evaluate(rouge_args=cmd)
    output_dict = r.output_to_dict(output)

    return output_dict, output
