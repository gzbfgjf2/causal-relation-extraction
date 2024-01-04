from src.data_process_scripts_parts.read import data_paths_to_data_dict


def create_txt_view(data_paths):
    data_dict = data_paths_to_data_dict(data_paths)
    with open('dataset/view.txt', 'w') as f:
        for v in data_dict.values():
            f.write(v.sentence + '\n')
            if len(v.causality_instances) == 0:
                f.write('\n\n\n')
                continue
            for i, instance in enumerate(v.causality_instances.values()):
                f.write('instance ' + str(i) + ': \n')
                f.write('cause: \n')
                for span in instance.cause.spans:
                    x, y = span
                    f.write(f'span: {x}:{y}\n')
                    f.write(f'text: {v.sentence[x:y]}')
                    f.write('\n')
                f.write('connective: \n')
                for span in instance.connective.spans:
                    x, y = span
                    f.write(f'span: {x}:{y}\n')
                    f.write(f'text: {v.sentence[x:y]}')
                    f.write('\n')
                f.write('effect: \n')
                for span in instance.effect.spans:
                    x, y = span
                    f.write(f'span: {x}:{y}\n')
                    f.write(f'text: {v.sentence[x:y]}')
                    f.write('\n')
                f.write('\n')
            f.write('\n')
